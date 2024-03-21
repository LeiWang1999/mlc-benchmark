import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build
import time


@tvm.register_func
def tvm_callback_cuda_postproc(code, _):
    code = code.replace(
        "default_function_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {",
        """default_function_kernel(half* __restrict__ A, half* __restrict__ B, half* __restrict__ C) {
  const int MAX_BLOCK_N = 8;
  const auto baseBlockIdx = blockIdx.x + gridDim.x *blockIdx.y;
  const auto totalPanel = (gridDim.x * gridDim.y +MAX_BLOCK_N * gridDim.x - 1) / (MAX_BLOCK_N * gridDim.x);
  const auto totalBlock = gridDim.x * gridDim.y;
  const auto panelIdx = baseBlockIdx / (MAX_BLOCK_N *gridDim.x);
  const auto strideLd = panelIdx + 1 < totalPanel ?MAX_BLOCK_N : (totalBlock - panelIdx * (MAX_BLOCK_N *gridDim.x)) / gridDim.x;
  const auto bx = (panelIdx & 1) ? gridDim.x -(baseBlockIdx - panelIdx * MAX_BLOCK_N * gridDim.x) /strideLd - 1 : (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) / strideLd;
  const auto by = (baseBlockIdx - panelIdx * MAX_BLOCK_N *gridDim.x) % strideLd + panelIdx * MAX_BLOCK_N;
  const auto bz = blockIdx.z;
  const dim3 blockIdx(bx, by, bz);
  """,
    )
    return code


def matmul_nt_propagate_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True, "smooth_b": True})
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N // 16, K // 16, 16, 16], dtype="float16")
            C = T.match_buffer(c, [M, N], dtype=in_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=out_dtype)

            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[vj // 16, vk // 16, vj % 16, vk % 16]

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float16(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B_reindex[vj, vk].astype(
                        out_dtype)
    return MyModule

def matmul_nt_propagate_a_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MyModule:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True, "smooth_a": True, "smooth_b": True})
            A = T.match_buffer(a, [M // 16, K // 16, 16, 16], dtype=in_dtype)
            B = T.match_buffer(b, [N // 16, K // 16, 16, 16], dtype="float16")
            C = T.match_buffer(c, [M, N], dtype=in_dtype)
            A_reindex = T.alloc_buffer([M, K], dtype=in_dtype)
            B_reindex = T.alloc_buffer([N, K], dtype=in_dtype)

            for i, k in T.grid(M, K):
                with T.block("A_reindex"):
                    vj, vk = T.axis.remap("SS", [i, k])
                    A_reindex[vj, vk] = A[vj // 16, vk // 16, vj % 16, vk % 16]
                    
            for j, k in T.grid(N, K):
                with T.block("B_reindex"):
                    vj, vk = T.axis.remap("SS", [j, k])
                    B_reindex[vj, vk] = B[vj // 16, vk // 16, vj % 16, vk % 16]

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float16(0)
                    C[vi, vj] = C[vi, vj] + A_reindex[vi, vk].astype(out_dtype) * B_reindex[vj, vk].astype(
                        out_dtype)
    return MyModule

benchmark_sets = [
    # (prim_func, input_args, default_dlight_schedule),
    (matmul_nt_propagate_b, (16384, 16384, 16384, "float16", "float16"), Matmul),
    (matmul_nt_propagate_a_b, (16384, 16384, 16384, "float16", "float16"), Matmul)
]
benchmark_results = {}
for get_prim_func, input_args, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    arch = CUDA(target)
    policy = DefaultPolicy(func=func, arch=arch)
    try:
        tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
    except:
        tags = None
    if tags:
        policy = TensorCorePolicy(func=tensorized_func, arch=arch, tags=tags)

    configs = policy.emit_config(20)
    for config in configs:
        print(config)

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    # print(best.code)
    # print(best.sch.mod)
    fast_tune_time = time.time() - tune_start
    print(
        "[FastDlight] The best latency of top 1 is {:.3f} ms".format(
            cpresults[0].latency
        )
    )
    print(
        "[FastDlight] The best latency of top 20 is {:.3f} ms".format(
            best.latency
        )
    )

    # evaluate the performance of the default schedule

    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
    with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
        mod_default = tvm.build(sch_default.mod["main"], target="cuda")
    default_tune_time = time.time() - default_tune_start

    args = func.buffer_map.values()

    profile_tensors = []
    for arg in args:
        profile_tensors.append(
            tvm.nd.array(
                np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype),
                device=arch.device,
            )
        )

    timer_cuda_mod = mod_default.time_evaluator(
        mod_default.entry_name, arch.device, number=5
    )
    t = timer_cuda_mod(*profile_tensors).mean

    print("Time cost of Dlight default schedule: {:.3f} ms".format(t * 1e3))

    profile_config = {
        f"{get_prim_func.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "fast_dlight_top20_tune_time": fast_tune_time,
            "fast_dlight_top1_latency": cpresults[0].latency,
            "fast_dlight_top20_latency": best.latency,
            "default_dlight_tune_time": default_tune_time,
            "default_dlight_latency": t * 1e3,
        }
    }
    benchmark_results.update(profile_config)

headers = [
    "PrimFunc",
    "Input Arguments",
    "FastDLight Top20 Tune Time",
    "FastDLight Top1 Latency",
    "FastDLight Top20 Latency",
    "DefaultDLight Tune Time",
    "DefaultDLight Latency",
]

col_width = (
    max(len(word) for row in [headers] + list(profile_config.values()) for word in row)
    + 2
)  # padding

print("".join(word.ljust(col_width) for word in headers))

print("-" * col_width * len(headers))

for config, values in benchmark_results.items():
    args = config.split("-")
    func_name = args[0]
    input_args = "-".join(args[1:])
    row = [
        func_name,
        input_args,
        f" {str(values['fast_dlight_top20_tune_time'])} s",
        f"{values['fast_dlight_top1_latency']:.3f} ms",
        f"{values['fast_dlight_top20_latency']:.3f} ms",
        str(values["default_dlight_tune_time"]),
        f"{values['default_dlight_latency']:.3f} ms",
    ]
    print("".join(word.ljust(col_width) for word in row))
