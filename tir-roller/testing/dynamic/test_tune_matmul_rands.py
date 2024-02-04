import tvm
import time
import numpy as np
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import fast_tune, fast_tune_with_dynamic_range
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
    (matmul_nt, (1024, 1024, "float32", "float32")),
]

for get_prim_func, input_args in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    
    dispatch_mod = fast_tune_with_dynamic_range(func, target, topk=5, parallel_build=True, dynamic_range={"m": [32, 64]})
    print(dispatch_mod)
    
    test_m_dim = 1024
    test_k_dim = 1024
    test_n_dim = 1024
    dtype = "float32"
    a_np = np.random.rand(test_m_dim, test_k_dim).astype(dtype)
    b_np = np.random.rand(test_n_dim, test_k_dim).astype(dtype)
    c_np = np.zeros((test_m_dim, test_n_dim)).astype(dtype)
    a = tvm.nd.array(a_np, device=tvm.cuda())
    b = tvm.nd.array(b_np, device=tvm.cuda())
    c = tvm.nd.array(c_np, device=tvm.cuda())
    rt_mod = tvm.build(dispatch_mod, target="cuda")
    rt_mod(a, b, c)
    print(c)
    