from vllm._C import ops
import torch
import numpy as np

import time

shapes = [
    (1, 16384, 16384),
    (1, 43008, 14336),
    (1, 14336, 14336),
    (1, 57344, 14336),
    (1, 14336, 57344),
    (1, 9216, 9216),
    (1, 36864, 9216),
    (1, 9216, 36864),
    (1, 22016, 8192),
    (1, 8192, 22016),
    (1, 8192, 8192),
    (1, 28672, 8192),
    (1, 8192, 28672),
    (16384, 16384, 16384),
    (8192, 43008, 14336),
    (8192, 14336, 14336),
    (8192, 57344, 14336),
    (8192, 14336, 57344),
    (8192, 9216, 9216),
    (8192, 36864, 9216),
    (8192, 9216, 36864),
    (8192, 22016, 8192),
    (8192, 8192, 22016),
    (8192, 8192, 8192),
    (8192, 28672, 8192),
    (8192, 8192, 28672),
]
for M, N, K in shapes:
    groups = 128
    bits = 4
    pack_factor = 8
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    qweight = torch.randint(
        0, 255, (K, N // pack_factor), device="cuda", dtype=torch.int32
    )
    scales = torch.randn(K // groups, N, device="cuda", dtype=torch.float16)
    qzeros = torch.randint(
        0, 255, (K // groups, N // pack_factor), device="cuda", dtype=torch.int32
    )
    out = ops.awq_gemm(x, qweight, scales, qzeros, pack_factor)

    # benchmark
    # torch inductor
    def get_runtime():
        tic = time.time()
        _ = ops.awq_gemm(x, qweight, scales, qzeros, pack_factor)
        return (time.time() - tic) * 1000

    with torch.no_grad():
        st = time.time()
        while time.time() - st < 1.0:
            get_runtime()  # warmup
        times = [get_runtime() for i in range(100)]
        print(f"vllm llama run {M} {N} {K} avg: {np.mean(times)} ms")
        torch.cuda.synchronize()
