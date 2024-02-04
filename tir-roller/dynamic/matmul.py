import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import fast_tune_with_dynamic_range
import time

def matmul_nt(N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNT:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
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


def matmul_nn(N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNN:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            m = T.int32()
            A = T.match_buffer(a, [m, K], dtype=in_dtype)
            B = T.match_buffer(b, [K, N], dtype=in_dtype)
            C = T.match_buffer(c, [m, N], dtype=out_dtype)

            for i, j, k in T.grid(m, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = tvm.tir.const(0, out_dtype)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[
                        vk, vj
                    ].astype(out_dtype)

    return MatmulNN

benchmark_sets = [
    # (prim_func, input_args, default_dlight_schedule),
    # (matmul_nt, (1024, 1024, "float32", "float32"), Matmul),
    (matmul_nt, (8192, 8192, "float16", "float16"), Matmul),
]
benchmark_results = {}
for get_prim_func, input_args, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    mod = fast_tune_with_dynamic_range(func, target, topk=20, parallel_build=True, dynamic_range={"m": [32, 64, 128, 256, 512]})
    with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
        rt_mod = tvm.build(mod, target=target)

    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
    with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
        mod_default = tvm.build(sch_default.mod, target="cuda")
    default_tune_time = time.time() - default_tune_start

    args = func.buffer_map.values()

    # evaluate the performance of the tuned schedule
    N, K, input_dytype, output_dtype = input_args
    for m in [32, 64, 4096, 8192]:
        a_np = np.random.rand(m, K).astype(input_dytype)
        b_np = np.random.rand(N, K).astype(input_dytype)
        c_np = np.zeros((m, N)).astype(output_dtype)
        a = tvm.nd.array(a_np, device=tvm.cuda())
        b = tvm.nd.array(b_np, device=tvm.cuda())
        c = tvm.nd.array(c_np, device=tvm.cuda())
        fd_evaluator = rt_mod.time_evaluator(rt_mod.entry_name, tvm.cuda(), number=10)
        d_evaluator = mod_default.time_evaluator(mod_default.entry_name, tvm.cuda(), number=10)
        fd_time = fd_evaluator(a, b, c).mean
        d_time = d_evaluator(a, b, c).mean
        print(f"m = {m}, fd_time = {fd_time * 1e3}, d_time = {d_time * 1e3}")
