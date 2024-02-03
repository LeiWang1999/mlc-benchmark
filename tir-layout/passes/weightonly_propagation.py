from typing import Dict, List, Optional, Set, Tuple, Union
import tvm
from tvm.relax.dpl import (
    is_op,
    rewrite_call,
    wildcard,
    TuplePattern,
    GlobalVarPattern,
)
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass
from tvm import relax
from tvm.relax.transform import function_pass
from tvm.relax import Expr
from tvm import tir
from enum import Enum
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import IterVar, PrimExpr, Var
from tvm.dlight.base import analysis
from tvm.tir.schedule.schedule import BlockRV
from tvm.relax import PyExprMutator
from tvm.relax.expr import Call
from tvm.dlight.gpu.matmul_analysis import (
    get_tensorized_func_and_tags,
    get_propagate_map,
)


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


class TransformKind(Enum):
    NonTransform = 0
    InterWarpTransform = 1
    IntraWarpTransform = 2


@function_pass(opt_level=0)
class InsertLayoutTransformToMatmul:
    def __init__(self, propogate_candidates):
        self.propogate_candidates = propogate_candidates
        self.g_var = GlobalVarPattern("dense")
        self.inp = wildcard()
        self.weight = wildcard()
        self.pattern = is_op("relax.call_tir")(
            self.g_var, TuplePattern([self.inp, self.weight]), add_constraint=False
        )
        self._new_exprs = []

    def transform_function(
        self, func: Expr, mod: IRModule, ctx: PassContext
    ) -> IRModule:
        updated_func = func

        def rewriter(expr, matches):
            if isinstance(expr, tvm.relax.expr.Call):
                g_var = matches[self.g_var]
                if g_var not in self.propogate_candidates.keys():
                    return expr
                if expr in self._new_exprs:
                    return expr
                else:
                    if len(self._new_exprs) != 0:
                        raise RuntimeError(
                            "Only one call_tir is allowed in the function"
                        )
                inp = matches[self.inp]
                weight = matches[self.weight]
                weight = relax.op.layout_transform(
                    weight, index_map=lambda i, j: (i, j)
                )
                new_expr = relax.call_tir(
                    g_var, (inp, weight), out_sinfo=expr.struct_info
                )
                self._new_exprs.append(new_expr)
                return new_expr
            return expr

        updated_func = rewrite_call(self.pattern, rewriter, func)

        return updated_func


def check_sm_version(arch: str) -> int:
    sm_version = arch.replace("sm_", "")
    return int(sm_version) if sm_version.isdigit() else -1


def get_in_out_dtypes(block: tir.Block) -> Tuple[str]:
    """
    Detect In/Out data types for the given block based on the analysis if read/write buffers.
    """
    assert len(block.reads) > 0 and len(block.writes) > 0
    in_dtype = block.reads[0].buffer.dtype
    out_dtype = block.writes[0].buffer.dtype
    return (in_dtype, out_dtype)


@module_pass(opt_level=0, name="InsertLayoutTransform")
class WeightOnlyLayoutPropagation:
    def __init__(
        self,
        transform_level: Union[int, TransformKind] = TransformKind.InterWarpTransform,
        target: Optional[Target] = None,
        faster_conversion: bool = False,
    ) -> None:
        if isinstance(transform_level, int):
            transform_level = TransformKind(transform_level)
        assert transform_level in [
            TransformKind.InterWarpTransform,
            TransformKind.IntraWarpTransform,
        ]
        # transform_level 1: only transform the inter-warp memory layout
        # transform_level 2: transform the inter-warp memory layout and the intra-warp memory layout
        self.transform_level = transform_level
        self.target = Target.current() if target is None else target
        # fast type conversion on nvidia gpu also requires weight permutation
        self.faster_conversion = faster_conversion

    def detect_propagate_matmul(self, func: tir.PrimFunc, target: Target):
        _, tags = get_tensorized_func_and_tags(func, self.target)
        if tags is None:
            return False, None
        return True, tags["intrin_info"]

    def transform_matmul(self, func: tir.PrimFunc, intrin_info):
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_mma_intrin_group,
        )

        sch = tir.Schedule(func)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None or len(reduction_blocks) != 1:
            return False
        (main_block,) = reduction_blocks

        intrin_group = get_mma_intrin_group(
            load_scope="shared",
            store_scope="shared",
            in_dtype=intrin_info["in_dtype"],
            out_dtype=intrin_info["out_dtype"],
            trans_a=False,
            trans_b=intrin_info["trans_b"],
        )

        _, inter_j, inter_k = intrin_group["micro_kernel"]

        sch.transform_layout(
            main_block,
            ("read", 1),
            lambda i, j: (i // inter_j, j // inter_k, i % inter_j, j % inter_k),
        )

        if self.transform_level.value >= TransformKind.IntraWarpTransform.value:
            permutation, _ = get_propagate_map(intrin_info["trans_b"])
            block = sch.reindex(main_block, ("read", 1))
            sch.transform_layout(block, ("read", 0), permutation)

        return sch.mod["main"]

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        from tvm.tir.tensor_intrin.cuda import (  # pylint: disable=import-outside-toplevel
            get_mma_intrin_group,
        )

        if self.target.kind.name != "cuda":
            # currently weight propagation only support nvidia gpus
            return mod

        propogate_candidates = {}
        candidates_intrin_info = {}
        for g_var, func in mod.functions_items():
            if g_var.name_hint != "main":
                # Note: this can be applied to any function which can be transformed to matmul (e.g., conv2d)
                # for mlc we only consider matmul
                # detect the pattern
                is_matmul, intrin_info = self.detect_propagate_matmul(func, self.target)

                if (
                    func.attrs is not None
                    and "dlight.do_not_tensorize" in func.attrs.keys()
                ):
                    # currently we only support tensorize propagation
                    continue

                if is_matmul:
                    if intrin_info["in_dtype"] != "float16":
                        # currently we only support float16 mma.
                        continue
                    # lift tags to the function as it has intrinsic information that can be reused.
                    propogate_candidates[g_var] = func
                    candidates_intrin_info[g_var] = intrin_info

        for g_var, func in propogate_candidates.items():
            updated_func = self.transform_matmul(
                func, candidates_intrin_info[g_var]
            ).with_attrs({"transform_kind": self.transform_level.value, "smooth_b": True})
            mod[g_var] = updated_func

        @relax.expr_functor.mutator
        class Mutator(PyExprMutator):
            """Mutator that performs transformation."""

            def __init__(self):
                super().__init__()
                self.transform_level = TransformKind.NonTransform

            def visit_call_(self, call_node: Call):
                g_var = call_node.args[0]
                if g_var not in propogate_candidates.keys():
                    return super().visit_call_(call_node)
                inp, weight = call_node.args[1]
                intrin_info = candidates_intrin_info[g_var]
                intrin_group = get_mma_intrin_group(
                    load_scope="shared",
                    store_scope="shared",
                    in_dtype=intrin_info["in_dtype"],
                    out_dtype=intrin_info["out_dtype"],
                    trans_a=False,
                    trans_b=intrin_info["trans_b"],
                )
                _, inter_j, inter_k = intrin_group["micro_kernel"]
                bl, br = (
                    (inter_j, inter_k) if intrin_info["trans_b"] else (inter_k, inter_j)
                )
                weight = self.builder_.emit(
                    relax.op.layout_transform(
                        weight,
                        index_map=lambda i, j: (i // bl, j // br, i % bl, j % br),
                    )
                )
                if self.transform_level.value >= TransformKind.IntraWarpTransform.value:
                    permutation, _ = get_propagate_map(candidates_intrin_info[g_var])
                    weight = self.builder_.emit(
                        relax.op.layout_transform(weight, index_map=permutation)
                    )
                call_node = self.builder_.emit(
                    relax.call_tir(
                        g_var, (inp, weight), out_sinfo=call_node.struct_info
                    )
                )
                return call_node

            def transform(
                self, transform_level: TransformKind = TransformKind.InterWarpTransform
            ):
                self.transform_level = transform_level
                for gv, func in mod.functions_items():
                    if isinstance(func, relax.Function):
                        updated_func = self.visit_expr(func)
                        self.builder_.update_func(gv, updated_func)
                new_mod = self.builder_.get()
                new_mod = new_mod.with_attrs(mod.attrs) if mod.attrs else new_mod
                return new_mod

        mod["main"] = Mutator().transform(self.transform_level)["main"]

        mod = relax.transform.LegalizeOps()(mod)
        return mod
