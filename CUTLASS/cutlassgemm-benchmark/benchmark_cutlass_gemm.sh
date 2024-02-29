#!/bin/sh

export CUDA_VISIBLE_DEVICES=3

# cutlass_profiler  --operation=Gemm \
#    --m=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --n=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --k=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --warmup-iterations=5 --profiling-iterations=10 \
#    --providers=cutlass --output=cutlass_gemm_performance.csv

python3 cutlass-gemm-bench.py | tee cutlass-gemm-bench_execution.log

# # single precision benchmark
# cutlass_profiler  --operation=Gemm \
#    --m=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --n=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --k=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --warmup-iterations=5 --profiling-iterations=10 \
#    --kernels=cutlass_simt_sgemm_128x128_8x2_* \
#    --providers=cutlass --output=cutlass_single_cudacore_performance.csv

# # half precision benchmark
# cutlass_profiler  --operation=Gemm \
#    --m=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --n=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --k=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --warmup-iterations=5 --profiling-iterations=10 \
#    --kernels=cutlass_simt_hgemm_256x128_8x2_* \
#    --providers=cutlass --output=cutlass_half_cudacore_performance.csv


# single precision benchmark
# cutlass_profiler  --operation=Gemm \
#    --m=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --n=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --k=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --warmup-iterations=5 --profiling-iterations=10 \
#    --kernels=cutlass_simt_sgemm_128x128_8x2_* \
#    --providers=cutlass --output=cutlass_single_tensorcore_performance.csv

# half precision benchmark
# cutlass_profiler  --operation=Gemm \
#    --m=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --n=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --k=2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384 \
#    --warmup-iterations=5 --profiling-iterations=10 \
#    --kernels=cutlass_tensorop_s*gemm_f16_*_nt_align \
#    --providers=cutlass --output=cutlass_half_tensorcore_performance.csv

# cutlass_profiler  --operation=Gemm \
#    --m=65536,128 \
#    --k=2,4032,1024,2048,4096,30522 \
#    --n=1000,1024,4096 \
#    --warmup-iterations=5 --profiling-iterations=10 \
#    --kernels=cutlass_simt_sgemm_128x128_8x2_* \
#    --providers=cutlass --output=cutlass_roller_single_cudacore_performance.csv

# cutlass_profiler  --operation=Gemm \
#    --m=65536,128 \
#    --k=2,4032,1024,2048,4096,30522 \
#    --n=1000,1024,4096 \
#    --warmup-iterations=5 --profiling-iterations=10 \
#    --kernels=cutlass_simt_hgemm_256x128_8x2_* \
#    --providers=cutlass --output=cutlass_roller_half_cudacore_performance.csv

# cutlass_profiler  --operation=Gemm \
#    --m=65536,128 \
#    --k=2,4032,1024,2048,4096,30522 \
#    --n=1000,1024,4096 \
#    --warmup-iterations=5 --profiling-iterations=10 \
#    --kernels=cutlass_tensorop_s*gemm_f16_*_nt_align \
#    --providers=cutlass --output=cutlass_roller_half_tensorcore_performance.csv
