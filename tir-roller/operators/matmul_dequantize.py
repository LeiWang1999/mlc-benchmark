import numpy as np
import tvm
from tvm import te
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build
import time


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


def matmul_nt_i4_propagate_b(M, N, K, in_dtype="float16", out_dtype="float16"):
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
            'target_format': "float16"
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
            'fast_decoding': True,
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


def matmul_nt_af4(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask
        
    A = te.placeholder((M, K), name='A', dtype=in_dtype)
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    LUT = te.placeholder((1 << bit, ), name='LUT', dtype='float16')


    def decode_func(n, k):
        w = _tir_u8_to_int(bit, B[n, k // n_float_per_i8], k % n_float_per_i8)
        return LUT[w]

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
    func = te.create_prim_func([A, B, LUT, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'source_format':{
                'bits': 4,
                'format': 'af',
            },
            'target_format': "float16"
        }
    })
    return tvm.IRModule.from_expr(func)

def matmul_nt_af4_propagate_a_b(M, N, K, in_dtype="float16", out_dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return (val >> (pos * nbit).astype("int8")) & mask
    
    A = te.placeholder((M // 16, K // 16, 16, 16), name='A', dtype=in_dtype)
    B = te.placeholder((N // 16, K // 16, 16, 16 // 8 * bit), name='B', dtype='int8')
    LUT = te.placeholder((1 << bit, ), name='LUT', dtype='float16')

    def decode_func(n, k, nn, kk):
        w = _tir_u8_to_int(bit, B[n, k, nn, kk // n_float_per_i8], kk % n_float_per_i8)
        return LUT[w]

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
    func = te.create_prim_func([A, B, LUT, C]).with_attr("dequantize_info", {
        'B': {
            'decode_block': 'B_decode',
            'source_format':{
                'bits': 4,
                'format': 'af',
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
    (matmul_nt_i4_propagate_b, (16384, 16384, 16384, "float16", "float16"), Matmul),
    # (matmul_nt_i4_propagate_a_b, (16384, 16384, 16384, "float16", "float16"), Matmul),
    # (matmul_nt_af4, (16384, 16384, 16384, "float16", "float16"), Matmul),
    # (matmul_nt_af4_propagate_a_b, (16384, 16384, 16384, "float16", "float16"), Matmul),
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

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)
    print(best.code)
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
