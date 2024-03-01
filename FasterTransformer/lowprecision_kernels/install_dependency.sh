#!/bin/bash

git clone --recursive https://github.com/apache/tvm ft_tvm
cd ft_tvm
git checkout 1afbf20
git submodule update --init --recursive
mkdir build
cd build
cp ../config.cmake .
echo "set(USE_CUDA ON)" >> config.cmake
echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUBLAS ON)" >> config.cmake
echo "set(USE_CUTLASS ON)" >> config.cmake
cmake ..
make -j

cd ../..

export PYTHONPATH=$(pwd)/ft_tvm/python
