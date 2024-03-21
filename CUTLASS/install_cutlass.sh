NVCC_CUDA_ARCHS='89' # ADA
git clone https://github.com/NVIDIA/cutlass --recursive
cd cutlass
mkdir -p build
cd build
cmake .. -DCUTLASS_NVCC_ARCHS=$NVCC_CUDA_ARCHS -DCUTLASS_LIBRARY_KERNELS=all -DCUTLASS_LIBRARY_OPERATIONS=gemm -DCUTLASS_LIBRARY_IGNORE_KERNELS=complex
make cutlass_profiler -j16
cd ..
export PATH=$PWD/build/tools/profiler:$PATH