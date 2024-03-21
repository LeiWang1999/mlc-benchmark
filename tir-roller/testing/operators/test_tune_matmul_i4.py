import numpy as np
import tvm
from tvm import te
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build
import time

@tvm.register_func
def tvm_callback_cuda_postproc(code, _):
    code = code.replace(
        "half* __restrict__ C) {",
        """half* __restrict__ C) {
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

def matmul_nt(M, N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNT:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[vj, vk].astype(
                        out_dtype
                    )

    return MatmulNT


def matmul_nt_i4(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M, K), name='A', dtype=in_dtype)
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k):
        w = _tir_u8_to_int_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, dtype=in_dtype)
        return w

    B_decode = te.compute(
        (N, K),
        decode_func,
        name='B_decode'
    )

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'fast_decoding': True,
            'source_format':{
                'bits': 4,
                'format': 'int',
            },
            'target_format': "float16"
        }
    })
    return tvm.IRModule.from_expr(func)


benchmark_sets = [
    # (prim_func, input_args, fast_dlight_schedule, default_dlight_schedule),
    # (matmul_nt, (1024, 1024, 1024, "float16", "float16"), Matmul, Matmul),
    # (matmul_nt_i4, (1024, 1024, 1024, "float16", "float16"), Matmul, Matmul),
    # (matmul_nt, (16384, 16384, 16384, "float16", "float16"), Matmul, Matmul),
    (matmul_nt_i4, (16384, 16384, 16384, "float16", "float16"), Matmul, Matmul),
]
benchmark_results = {}
for get_prim_func, input_args, f_schedule, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    arch = CUDA(target)
    policy = TensorCorePolicy(
        func=func,
        arch=arch,
        tags={
            "tensorcore_config": [0, 1],
            "pipeline_stage": 2,
            "use_async_copy": 1,
        },
    )
    configs = policy.emit_config(1)
    for config in configs:
        print(config)
    rule = f_schedule()

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
    fast_tune_time = time.time() - tune_start
    print("[FastDlight] The best latency of top 1 is {:.3f} ms".format(cpresults[0].latency))
    print("[FastDlight] The best latency of top 20 is {:.3f} ms".format(best.latency))