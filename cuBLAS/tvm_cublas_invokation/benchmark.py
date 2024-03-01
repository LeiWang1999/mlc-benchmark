import tvm
from tvm import relay
from tvm import te
from tvm.contrib import cublas
import numpy as np

def relay_cublas_matmul_benckmark(m, n, k, in_dtype, out_dtype, transpose_a, transpose_b):
    input_shape = (k, m) if transpose_a else (m, k)
    weight_shape = (n, k) if transpose_b else (k, n)
    
    A = te.placeholder(input_shape, name="A", dtype=in_dtype)
    B = te.placeholder(weight_shape, name="B", dtype=in_dtype)
    
    C = cublas.matmul(A, B, dtype=out_dtype,
                      transa=transpose_a, transb=transpose_b)
    s = te.create_schedule(C.op)
    func = tvm.build(s, [A, B, C], "cuda")
    # time evaluation
    dev = tvm.device("cuda", 0)
    cuda_a = tvm.nd.array(np.random.uniform(0, 1, input_shape).astype(in_dtype), dev)
    cuda_b = tvm.nd.array(np.random.uniform(0, 1, weight_shape).astype(in_dtype), dev)
    cuda_c = tvm.nd.array(np.empty((m, n), dtype=out_dtype), dev)
    evaluator = func.time_evaluator(func.entry_name, dev, number=100)

    evaluator(cuda_a, cuda_b, cuda_c)
    
    num_flops = 2 * m * n * k

    num_runs = 5
    timer_cuda_mod = func.time_evaluator(
        func.entry_name, dev, number=num_runs)

    t = timer_cuda_mod(cuda_a, cuda_b, cuda_c).mean
    t = t * 1e3
    GFLOPS = num_flops / t / 1e6
    return t, GFLOPS


llm_shape =  [
    # (16384, 16384, 16384, "float16", "float16", False, False),
    # (8192, 43008, 14336, "float16", "float16", False, False),
    # (8192, 14336, 14336, "float16", "float16", False, False),
    # (8192, 57344, 14336, "float16", "float16", False, False),
    # (8192, 14336, 57344, "float16", "float16", False, False),
    # (8192, 9216, 9216, "float16", "float16", False, False),
    # (8192, 36864, 9216, "float16", "float16", False, False),
    # (8192, 9216, 36864, "float16", "float16", False, False),
    # (8192, 22016, 8192, "float16", "float16", False, False),
    # (8192, 8192, 22016, "float16", "float16", False, False),
    # (8192, 8192, 8192, "float16", "float16", False, False),
    # (8192, 28672, 8192, "float16", "float16", False, False),
    # (8192, 8192, 22016, "float16", "float16", False, False),
    
    # (16384, 16384, 16384, "float16", "float16", False, True),
    # (8192, 43008, 14336, "float16", "float16", False, True),
    # (8192, 14336, 14336, "float16", "float16", False, True),
    # (8192, 57344, 14336, "float16", "float16", False, True),
    # (8192, 14336, 57344, "float16", "float16", False, True),
    # (8192, 9216, 9216, "float16", "float16", False, True),
    # (8192, 36864, 9216, "float16", "float16", False, True),
    # (8192, 9216, 36864, "float16", "float16", False, True),
    # (8192, 22016, 8192, "float16", "float16", False, True),
    # (8192, 8192, 22016, "float16", "float16", False, True),
    # (8192, 8192, 8192, "float16", "float16", False, True),
    # (8192, 28672, 8192, "float16", "float16", False, True),
    # (8192, 8192, 22016, "float16", "float16", False, True),
    
    # (16384, 16384, 16384, "int8", "int32", False, True),
    # (8192, 43008, 14336, "int8", "int32", False, True),
    # (8192, 14336, 14336, "int8", "int32", False, True),
    # (8192, 57344, 14336, "int8", "int32", False, True),
    # (8192, 14336, 57344, "int8", "int32", False, True),
    # (8192, 9216, 9216, "int8", "int32", False, True),
    # (8192, 36864, 9216, "int8", "int32", False, True),
    # (8192, 9216, 36864, "int8", "int32", False, True),
    # (8192, 22016, 8192, "int8", "int32", False, True),
    # (8192, 8192, 22016, "int8", "int32", False, True),
    # (8192, 8192, 8192, "int8", "int32", False, True),
    # (8192, 28672, 8192, "int8", "int32", False, True),
    # (8192, 8192, 22016, "int8", "int32", False, True),
    
    (8192, 512, 512, "float16", "float16", False, True),
    (8192, 1536, 512, "float16", "float16", False, True),
    
]

int4_shape = [
    (1,	16384,	16384, "float16", "float16", False, True),
    (1,	43008,	14336, "float16", "float16", False, True),
    (1,	14336,	14336, "float16", "float16", False, True),
    (1,	57344,	14336, "float16", "float16", False, True),
    (1,	14336,	57344, "float16", "float16", False, True),
    (1,	9216,	9216, "float16", "float16", False, True),
    (1,	36864,	9216, "float16", "float16", False, True),
    (1,	9216,	36864, "float16", "float16", False, True),
    (1,	22016,	8192, "float16", "float16", False, True),
    (1,	8192,	22016, "float16", "float16", False, True),
    (1,	8192,	8192, "float16", "float16", False, True),
    (1,	28672,	8192, "float16", "float16", False, True),
    (1,	8192,	22016, "float16", "float16", False, True),
    (16,	16384,	16384, "float16", "float16", False, True),
    (16,	43008,	14336, "float16", "float16", False, True),
    (16,	14336,	14336, "float16", "float16", False, True),
    (16,	57344,	14336, "float16", "float16", False, True),
    (16,	14336,	57344, "float16", "float16", False, True),
    (16,	9216,	9216, "float16", "float16", False, True),
    (16,	36864,	9216, "float16", "float16", False, True),
    (16,	9216,	36864, "float16", "float16", False, True),
    (16,	22016,	8192, "float16", "float16", False, True),
    (16,	8192,	22016, "float16", "float16", False, True),
    (16,	8192,	8192, "float16", "float16", False, True),
    (16,	28672,	8192, "float16", "float16", False, True),
    (16,	8192,	22016, "float16", "float16", False, True),
    (32,	16384,	16384, "float16", "float16", False, True),
    (32,	43008,	14336, "float16", "float16", False, True),
    (32,	14336,	14336, "float16", "float16", False, True),
    (32,	57344,	14336, "float16", "float16", False, True),
    (32,	14336,	57344, "float16", "float16", False, True),
    (32,	9216,	9216, "float16", "float16", False, True),
    (32,	36864,	9216, "float16", "float16", False, True),
    (32,	9216,	36864, "float16", "float16", False, True),
    (32,	22016,	8192, "float16", "float16", False, True),
    (32,	8192,	22016, "float16", "float16", False, True),
    (32,	8192,	8192, "float16", "float16", False, True),
    (32,	28672,	8192, "float16", "float16", False, True),
    (32,	8192,	22016, "float16", "float16", False, True),
    (64,	16384,	16384, "float16", "float16", False, True),
    (64,	43008,	14336, "float16", "float16", False, True),
    (64,	14336,	14336, "float16", "float16", False, True),
    (64,	57344,	14336, "float16", "float16", False, True),
    (64,	14336,	57344, "float16", "float16", False, True),
    (64,	9216,	9216, "float16", "float16", False, True),
    (64,	36864,	9216, "float16", "float16", False, True),
    (64,	9216,	36864, "float16", "float16", False, True),
    (64,	22016,	8192, "float16", "float16", False, True),
    (64,	8192,	22016, "float16", "float16", False, True),
    (64,	8192,	8192, "float16", "float16", False, True),
    (64,	28672,	8192, "float16", "float16", False, True),
    (64,	8192,	22016, "float16", "float16", False, True),
    (128,	16384,	16384, "float16", "float16", False, True),
    (128,	43008,	14336, "float16", "float16", False, True),
    (128,	14336,	14336, "float16", "float16", False, True),
    (128,	57344,	14336, "float16", "float16", False, True),
    (128,	14336,	57344, "float16", "float16", False, True),
    (128,	9216,	9216, "float16", "float16", False, True),
    (128,	36864,	9216, "float16", "float16", False, True),
    (128,	9216,	36864, "float16", "float16", False, True),
    (128,	22016,	8192, "float16", "float16", False, True),
    (128,	8192,	22016, "float16", "float16", False, True),
    (128,	8192,	8192, "float16", "float16", False, True),
    (128,	28672,	8192, "float16", "float16", False, True),
    (128,	8192,	22016, "float16", "float16", False, True),
    (256,	16384,	16384, "float16", "float16", False, True),
    (256,	43008,	14336, "float16", "float16", False, True),
    (256,	14336,	14336, "float16", "float16", False, True),
    (256,	57344,	14336, "float16", "float16", False, True),
    (256,	14336,	57344, "float16", "float16", False, True),
    (256,	9216,	9216, "float16", "float16", False, True),
    (256,	36864,	9216, "float16", "float16", False, True),
    (256,	9216,	36864, "float16", "float16", False, True),
    (256,	22016,	8192, "float16", "float16", False, True),
    (256,	8192,	22016, "float16", "float16", False, True),
    (256,	8192,	8192, "float16", "float16", False, True),
    (256,	28672,	8192, "float16", "float16", False, True),
    (256,	8192,	22016, "float16", "float16", False, True),
    (512,	16384,	16384, "float16", "float16", False, True),
    (512,	43008,	14336, "float16", "float16", False, True),
    (512,	14336,	14336, "float16", "float16", False, True),
    (512,	57344,	14336, "float16", "float16", False, True),
    (512,	14336,	57344, "float16", "float16", False, True),
    (512,	9216,	9216, "float16", "float16", False, True),
    (512,	36864,	9216, "float16", "float16", False, True),
    (512,	9216,	36864, "float16", "float16", False, True),
    (512,	22016,	8192, "float16", "float16", False, True),
    (512,	8192,	22016, "float16", "float16", False, True),
    (512,	8192,	8192, "float16", "float16", False, True),
    (512,	28672,	8192, "float16", "float16", False, True),
    (512,	8192,	22016, "float16", "float16", False, True),
    (1024,	16384,	16384, "float16", "float16", False, True),
    (1024,	43008,	14336, "float16", "float16", False, True),
    (1024,	14336,	14336, "float16", "float16", False, True),
    (1024,	57344,	14336, "float16", "float16", False, True),
    (1024,	14336,	57344, "float16", "float16", False, True),
    (1024,	9216,	9216, "float16", "float16", False, True),
    (1024,	36864,	9216, "float16", "float16", False, True),
    (1024,	9216,	36864, "float16", "float16", False, True),
    (1024,	22016,	8192, "float16", "float16", False, True),
    (1024,	8192,	22016, "float16", "float16", False, True),
    (1024,	8192,	8192, "float16", "float16", False, True),
    (1024,	28672,	8192, "float16", "float16", False, True),
    (1024,	8192,	22016, "float16", "float16", False, True),
    (2048,	16384,	16384, "float16", "float16", False, True),
    (2048,	43008,	14336, "float16", "float16", False, True),
    (2048,	14336,	14336, "float16", "float16", False, True),
    (2048,	57344,	14336, "float16", "float16", False, True),
    (2048,	14336,	57344, "float16", "float16", False, True),
    (2048,	9216,	9216, "float16", "float16", False, True),
    (2048,	36864,	9216, "float16", "float16", False, True),
    (2048,	9216,	36864, "float16", "float16", False, True),
    (2048,	22016,	8192, "float16", "float16", False, True),
    (2048,	8192,	22016, "float16", "float16", False, True),
    (2048,	8192,	8192, "float16", "float16", False, True),
    (2048,	28672,	8192, "float16", "float16", False, True),
    (2048,	8192,	22016, "float16", "float16", False, True),
    (4096,	16384,	16384, "float16", "float16", False, True),
    (4096,	43008,	14336, "float16", "float16", False, True),
    (4096,	14336,	14336, "float16", "float16", False, True),
    (4096,	57344,	14336, "float16", "float16", False, True),
    (4096,	14336,	57344, "float16", "float16", False, True),
    (4096,	9216,	9216, "float16", "float16", False, True),
    (4096,	36864,	9216, "float16", "float16", False, True),
    (4096,	9216,	36864, "float16", "float16", False, True),
    (4096,	22016,	8192, "float16", "float16", False, True),
    (4096,	8192,	22016, "float16", "float16", False, True),
    (4096,	8192,	8192, "float16", "float16", False, True),
    (4096,	28672,	8192, "float16", "float16", False, True),
    (4096,	8192,	22016, "float16", "float16", False, True),
    (8192,	16384,	16384, "float16", "float16", False, True),
    (8192,	43008,	14336, "float16", "float16", False, True),
    (8192,	14336,	14336, "float16", "float16", False, True),
    (8192,	57344,	14336, "float16", "float16", False, True),
    (8192,	14336,	57344, "float16", "float16", False, True),
    (8192,	9216,	9216, "float16", "float16", False, True),
    (8192,	36864,	9216, "float16", "float16", False, True),
    (8192,	9216,	36864, "float16", "float16", False, True),
    (8192,	22016,	8192, "float16", "float16", False, True),
    (8192,	8192,	22016, "float16", "float16", False, True),
    (8192,	8192,	8192, "float16", "float16", False, True),
    (8192,	28672,	8192, "float16", "float16", False, True),
    (8192,	8192,	22016, "float16", "float16", False, True),
    (16384,	16384,	16384, "float16", "float16", False, True),
    (16384,	43008,	14336, "float16", "float16", False, True),
    (16384,	14336,	14336, "float16", "float16", False, True),
    (16384,	57344,	14336, "float16", "float16", False, True),
    (16384,	14336,	57344, "float16", "float16", False, True),
    (16384,	9216,	9216, "float16", "float16", False, True),
    (16384,	36864,	9216, "float16", "float16", False, True),
    (16384,	9216,	36864, "float16", "float16", False, True),
    (16384,	22016,	8192, "float16", "float16", False, True),
    (16384,	8192,	22016, "float16", "float16", False, True),
    (16384,	8192,	8192, "float16", "float16", False, True),
    (16384,	28672,	8192, "float16", "float16", False, True),
    (16384,	8192,	22016, "float16", "float16", False, True),
]


int1_shape = [
    (1, 1024, 8192, "int8", "int8", False, True),
    (1, 8192, 8192, "int8", "int8", False, True),
    (1, 8192, 28672, "int8", "int8", False, True),
    (1, 28672, 8192, "int8", "int8", False, True),
    (16, 1024, 8192, "int8", "int8", False, True),
    (16, 8192, 8192, "int8", "int8", False, True),
    (16, 8192, 28672, "int8", "int8", False, True),
    (16, 28672, 8192, "int8", "int8", False, True),
    (32, 1024, 8192, "int8", "int8", False, True),
    (32, 8192, 8192, "int8", "int8", False, True),
    (32, 8192, 28672, "int8", "int8", False, True),
    (32, 28672, 8192, "int8", "int8", False, True),
    (64, 1024, 8192, "int8", "int8", False, True),
    (64, 8192, 8192, "int8", "int8", False, True),
    (64, 8192, 28672, "int8", "int8", False, True),
    (64, 28672, 8192, "int8", "int8", False, True),
    (128, 1024, 8192, "int8", "int8", False, True),
    (128, 8192, 8192, "int8", "int8", False, True),
    (128, 8192, 28672, "int8", "int8", False, True),
    (128, 28672, 8192, "int8", "int8", False, True),
    (1024, 1024, 8192, "int8", "int8", False, True),
    (1024, 8192, 8192, "int8", "int8", False, True),
    (1024, 8192, 28672, "int8", "int8", False, True),
    (1024, 28672, 8192, "int8", "int8", False, True),
    (4096, 1024, 8192, "int8", "int8", False, True),
    (4096, 8192, 8192, "int8", "int8", False, True),
    (4096, 8192, 28672, "int8", "int8", False, True),
    (4096, 28672, 8192, "int8", "int8", False, True),
    (8192, 1024, 8192, "int8", "int8", False, True),
    (8192, 8192, 8192, "int8", "int8", False, True),
    (8192, 8192, 28672, "int8", "int8", False, True),
    (8192, 28672, 8192, "int8", "int8", False, True),
    (16384, 1024, 8192, "int8", "int8", False, True),
    (16384, 8192, 8192, "int8", "int8", False, True),
    (16384, 8192, 28672, "int8", "int8", False, True),
    (16384, 28672, 8192, "int8", "int8", False, True),

]
for llm in int1_shape:
    t, gflops = relay_cublas_matmul_benckmark(*llm)
    print("time: {} ms, gflops: {}".format(t, gflops))
    

    

