import numpy as np
import tvm
from tvm.script import tir as T
from tvm import te
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags
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

def matmul_nt_i4(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M, K), name='A', dtype=in_dtype)
    B = te.placeholder((N // 16, K // 16, 16, 16 // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k, nn, kk):
        w = _tir_u8_to_int_to_float(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, dtype=in_dtype)
        return w

    B_decode = te.compute(
        (N // 16, K // 16, 16, 16),
        decode_func,
        name='B_decode'
    )
    
    B_reindex = te.compute(
        (N, K),
        lambda i, j: B_decode[i // 16, j // 16, i % 16, j % 16],
        name="B_reindex"
    )
    
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_reindex[j, k], axis=k),
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
            'target_format':{
                'bits': 16,
                'format': 'float',
            }
        }
    })
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)

def matmul_nt_i4_propagate_a_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M // 16, K // 16, 16, 16), name='A', dtype=in_dtype)
    B = te.placeholder((N // 16, K // 16, 16, 16 // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k, nn, kk):
        w = _tir_u8_to_int_to_float(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8, dtype=in_dtype)
        return w

    B_decode = te.compute(
        (N // 16, K // 16, 16, 16),
        decode_func,
        name='B_decode'
    )
    
    B_reindex = te.compute(
        (N, K),
        lambda i, j: B_decode[i // 16, j // 16, i % 16, j % 16],
        name="B_reindex"
    )
    
    A_reindex = te.compute(
        (M, K),
        lambda i, j: A[i // 16, j // 16, i % 16, j % 16],
        name="A_reindex"
    )
    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A_reindex[i, k] * B_reindex[j, k], axis=k),
        name='C'
    )
    func = te.create_prim_func([A, B, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'fast_decoding': False,
            'source_format':{
                'bits': 4,
                'format': 'int',
            },
            'target_format': "float16"
        }
    })
    func = func.with_attr("smooth_a", True)
    func = func.with_attr("smooth_b", True)

    return tvm.IRModule.from_expr(func)

benchmark_sets = [
    # (prim_func, input_args, default_dlight_schedule),
    # (matmul_nt_i4, (16384, 16384, 16384, "float16", "float16"), Matmul),
    (matmul_nt_i4_propagate_a_b, (16384, 16384, 16384, "float16", "float16"), Matmul)
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

    configs = policy.emit_config(1)
    for config in configs:
        print(config)

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
    # print(best.config)
    print(best.code)
    # print(best.sch.mod["main"])
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

