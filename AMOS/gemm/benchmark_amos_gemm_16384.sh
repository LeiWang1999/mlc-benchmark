export AMOS_HOME=/workspace/v-leiwang3/AMOS
export PYTHONPATH=$AMOS_HOME/python
export CUDA_VISIBLE_DEVICES=0,1,2,3 

# CUDA_VISIBLE_DEVICES=1,2,3 python3 /workspace/v-leiwang3/lowbit-benchmark/1.matmul_bench/amosgemm/test_gemm.py --trials 1000 --simple_mode 1 | tee amos_tunning_simple.log
python3 /workspace/v-leiwang3/lowbit-benchmark/1.matmul_bench/amosgemm/test_gemm_16384.py --trials 1000 --simple_mode 1 --in_dtype int8 --out_dtype int32 | tee amos_tunning_simple_int8_nn.log
# CUDA_VISIBLE_DEVICES=1,2,3 python3 /workspace/v-leiwang3/lowbit-benchmark/1.matmul_bench/amosgemm/test_gemm.py --trials 1000 --simple_mode 0 | tee amos_tunning_no_simple.log