#!/bin/bash

git clone --recursive https://github.com/apache/tvm trt_llm_tvm
cd trt_llm_tvm
git checkout 2bf3a0a4287069ac55ee3304c285b08592d3d1bc
git submodule update --init --recursive
mkdir build
cp ../config.cmake .
echo "set(USE_CUDA ON)" >> config.cmake
echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUBLAS ON)" >> config.cmake
echo "set(USE_CUTLASS ON)" >> config.cmake
cmake ..
make -j

cd ../..

export PYTHONPATH=(pwd)/trt_llm_tvm/python
