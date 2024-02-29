import tvm
from tvm import relay
from tvm import te
from tvm.contrib import cublaslt
import numpy as np

def test_relay_cublas_matmul(m, n, k, in_dtype, out_dtype, transpose_a, transpose_b):
    input_shape = (m, k) if transpose_a else (k, m)
    weight_shape = (k, n) if transpose_b else (n, k)
    
    A = te.placeholder(input_shape, name="A", dtype=in_dtype)
    B = te.placeholder(weight_shape, name="B", dtype=in_dtype)
    
    C = cublaslt.matmul(A, B, dtype=out_dtype, transa=False, transb=True)    
    s = te.create_schedule(C.op)
    func = tvm.build(s, [A, B, C], "cuda")
    # time evaluation
    dev = tvm.device("cuda", 0)
    cuda_a = tvm.nd.array(np.random.uniform(0, 1, input_shape).astype(in_dtype), dev)
    cuda_b = tvm.nd.array(np.random.uniform(0, 1, weight_shape).astype(in_dtype), dev)
    cuda_c = tvm.nd.array(np.empty((m, n), dtype=out_dtype), dev)
        
    print("m={}, n={}, k={}, in_dtype={}, out_dtype={}, transpose_a={}, transpose_b={}".format(m, n, k, in_dtype, out_dtype, transpose_a, transpose_b))
    evaluator = func.time_evaluator(func.entry_name, dev, number=100)

    evaluator(cuda_a, cuda_b, cuda_c)
    
    
test_relay_cublas_matmul(1024, 1024, 1024, "float16", "float16", False, False)

    

