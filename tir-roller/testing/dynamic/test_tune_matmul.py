import tvm
import time
import numpy as np
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import fast_tune
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags


def matmul_nt(N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNT:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            m = T.int32()
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)

            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                        vj, vk
                    ].astype(out_dtype)

    return MatmulNT


benchmark_sets = [
    # (prim_func, input_args, fast_dlight_schedule, default_dlight_schedule),
    (matmul_nt, (1024, 1024, "float32", "float32"), Matmul, Matmul),
]


def var_warpper(v, opt_shapes):
    if isinstance(v, tvm.tir.Var):
        assert v.name in opt_shapes
        return opt_shapes[v.name]
    elif isinstance(v, tvm.tir.IntImm):
        return v.value
    else:
        raise RuntimeError("Not supported type: ", type(v))


benchmark_results = {}
for get_prim_func, input_args, f_schedule, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"].with_attr({"opt_shapes": {"m": 32}})
    target = tvm.target.Target("nvidia/nvidia-a100")
    cpresults, best = fast_tune(func, target, topk=20, parallel_build=True)
    tune_start = time.time()

    fast_tune_time = time.time() - tune_start
    print(
        "[FastDlight] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency * 1e3
        )
    )
    print(
        "[FastDlight] The best latency of top 20 is {:.3f} ms".format(
            best.latency * 1e3
        )
    )
