import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R, tir as T
from tvm import tir
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass


@I.ir_module
class Before:
    @T.prim_func(private=True)
    def add(
        A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        Out: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
    ):
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                Out[vi, vj] = A[vi, vj] + T.float16(1.0)

    @T.prim_func(private=True)
    def take(
        A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        B: T.Buffer((T.int64(1),), "int32"),
        T_take: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
    ):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(4096)):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

    @T.prim_func(private=True)
    def add1(
        A: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
        Out: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
    ):
        for i, j in T.grid(T.int64(1), T.int64(4096)):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                Out[vi, vj] = A[vi, vj] + T.float16(2.0)

    @T.prim_func(private=True)
    def relu(
        A: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
        Relu: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
    ):
        for i, j in T.grid(T.int64(1), T.int64(4096)):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                Relu[vi, vj] = T.max(A[vi, vj], T.float16(0.0))

    @R.function
    def main(
        input_ids: R.Tensor((1,), dtype="int32"),
        input_embeds: R.Tensor((4096, 4096), dtype="float16"),
    ) -> R.Tensor((1, 4096), dtype="float16"):
        cls = Before
        with R.dataflow():
            gv: R.Tensor((1, 4096), dtype="float16") = cls.fused_func(input_ids, input_embeds)
            R.output(gv)
        return gv

    @R.function(private=True)
    def fused_func(
        input_ids: R.Tensor((1,), dtype="int32"),
        input_embeds: R.Tensor((4096, 4096), dtype="float16"),
    ) -> R.Tensor((1, 4096), dtype="float16"):
        R.func_attr({"Primitive": 1})
        cls = Before
        with R.dataflow():
            lv = R.call_tir(
                cls.add, (input_embeds,), out_sinfo=R.Tensor((4096, 4096), dtype="float16")
            )
            lv1 = R.call_tir(
                cls.take, (lv, input_ids), out_sinfo=R.Tensor((1, 4096), dtype="float16")
            )
            lv2 = R.call_tir(cls.add1, (lv1,), out_sinfo=R.Tensor((1, 4096), dtype="float16"))
            gv = R.call_tir(cls.relu, (lv2,), out_sinfo=R.Tensor((1, 4096), dtype="float16"))
            R.output(gv)
        return gv


relax_mod = Before
relax_mod = relax.transform.FuseTIR()(relax_mod)
print(relax_mod)


@module_pass(opt_level=0, name="ApplyFastTuning")
class Traverse_mod:  # pylint: disable=too-few-public-methods
    """A IRModule pass that applies a list of ScheduleRules to all PrimFuncs in the module."""

    def __init__(
        self,
    ):
        pass

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        for g_var, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc):
                sch = tvm.tir.Schedule(func)
                print("implement sch.get_sref(sch.get_block('add')).stmt")
                print(sch.get_sref(sch.get_block("add")).stmt)
                print("implement sch.get_sref(sch.get_block('add_1')).stmt")
                print(sch.get_sref(sch.get_block("add_1")).stmt)
        return mod


relax_mod = Traverse_mod()(relax_mod)
