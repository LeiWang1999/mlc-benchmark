import numpy as np
import tvm
import time
from tvm.script import tir as T
from tvm.dlight.gpu import GEMV
from tvm import te


def gemv_i4(M, N, K, dtype="float16"):
    bit = 4
    n_float_per_i8 = 8 // bit

    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
    
    A = te.placeholder((M, K), name='A', dtype=dtype)
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')
    
    def decode_func(n, k):
        w = _tir_u8_to_int_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, dtype=dtype)
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
    func = te.create_prim_func([A, B, C]).with_attr("inconsistent", {
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
    return tvm.IRModule.from_expr(func)
    
def gemv(
    M, N, K, dtype="float16"
):
    @tvm.script.ir_module
    class GEMV:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=dtype)
            B = T.match_buffer(b, [N, K], dtype=dtype)
            C = T.match_buffer(c, [M, N], dtype=dtype)
            
            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + \
                        A[vi, vk] * B[vj, vk]
    return GEMV

benchmark_sets = [
    # (prim_func, input_args, fast_dlight_schedule, default_dlight_schedule),
    (gemv, (1, 1024, 1024, "float16"), GEMV),
    # (gemv, (1, 8192, 8192, "float16"), GEMV),
    # (gemv, (1, 16384, 16384, "float16"), GEMV),
    # (gemv_i4, (1, 16384, 16384, "float16"), GEMV),
]
benchmark_results = {}
for get_prim_func, input_args, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")

    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
    mod_default = tvm.build(sch_default.mod["main"], target="cuda")
    default_tune_time = time.time() - default_tune_start

    args = func.buffer_map.values()
            
    profile_tensors = []
    for arg in args:
        profile_tensors.append(tvm.nd.array(
            np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype), device=tvm.cuda())
        )
                    
    timer_cuda_mod = mod_default.time_evaluator(
        mod_default.entry_name, tvm.cuda(), number=5)
    t = timer_cuda_mod(*profile_tensors).mean

    print("Time cost of Dlight default schedule: {:.3f} ms".format(t * 1e3))
