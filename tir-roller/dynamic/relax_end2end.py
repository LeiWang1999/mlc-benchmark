import numpy as np
import tvm
import tvm.dlight as dl
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import fast_tune_with_dynamic_range
from tvm.script import ir as I, tir as T, relax as R
from tvm import relax

target = tvm.target.Target("cuda")


@tvm.script.ir_module
class Module:
    # TIR function can handle different cases.
    @T.prim_func
    def addone(a: T.handle, b: T.handle) -> None:
        m = T.int64()
        A = T.match_buffer(a, (m, 16))
        B = T.match_buffer(b, (m, 16))
        for i, j in T.grid(m, 16):
            with T.block("addone"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] + T.float32(1)

    @R.function
    def main(c0: R.Tensor(("m", 16), "float32")):
        m = T.int64()
        cls = Module
        with R.dataflow():
            gv3 = relax.call_tir(cls.addone, (c0,), R.Tensor((m, 16), dtype="float32"))
            R.output(gv3)
        return gv3


relax_mod = Module

# with target:
#     relax_mod = dl.ApplyDefaultSchedule(  # pylint: disable=not-callable
#         dl.gpu.Matmul(),
#         dl.gpu.GEMV(),
#         dl.gpu.Reduction(),
#         dl.gpu.GeneralReduction(),
#         dl.gpu.Fallback(),
#     )(relax_mod)

dynamic_range = {
    "m": [1, 16, 32],
}

relax_mod = dl.ApplyFastTuning(
    topk=20,
    target=target,
    dynamic_range=dynamic_range,
)(relax_mod)
relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
relax_mod = relax.transform.RewriteDataflowReshape()(relax_mod)
print(relax_mod)
with tvm.transform.PassContext(config={"tir.use_async_copy": False}):
    ex = relax.build(relax_mod, target)


# device = tvm.cuda(0)
# vm = relax.VirtualMachine(ex, device)


# input_args = []
# input_args.append(tvm.nd.array(np.random.uniform(-1, 1, size=(32, 16)).astype("float32"), device))

# res = vm["main"](*input_args)

# print(res)