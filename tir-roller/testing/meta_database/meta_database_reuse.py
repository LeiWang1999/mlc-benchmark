import tvm
import tvm.testing
from tvm import relax
from tvm.script import ir as I, relax as R, tir as T
from tvm import tir
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass

from tvm import dlight as dl

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
    def add1(
        A: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
        Out: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
    ):
        for i, j in T.grid(T.int64(4096), T.int64(4096)):
            with T.block("add"):
                vi, vj = T.axis.remap("SS", [i, j])
                Out[vi, vj] = A[vi, vj] + T.float16(1.0)

    @R.function
    def main(
        input_embeds: R.Tensor((4096, 4096), dtype="float16"),
    ) -> R.Tensor((4096, 4096), dtype="float16"):
        R.func_attr({"Primitive": 1})
        cls = Before
        with R.dataflow():
            lv = R.call_tir(
                cls.add, (input_embeds,), out_sinfo=R.Tensor((4096, 4096), dtype="float16")
            )
            gv = R.call_tir(cls.add1, (lv,), out_sinfo=R.Tensor((4096, 4096), dtype="float16"))
            R.output(gv)
        return gv


relax_mod = Before
target = tvm.target.Target("nvidia/nvidia-a100")
relax_mod = dl.ApplyFastTuning(topk=10, target=target, parallel_build=True, meta_database_dir="./e2e_mlp")(relax_mod)
