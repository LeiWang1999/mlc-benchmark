export AMOS_HOME=/workspace/v-leiwang3/AMOS
export PYTHONPATH=$AMOS_HOME/python

# CUDA_VISIBLE_DEVICES=1,2,3 python3 /workspace/v-leiwang3/lowbit-benchmark/2.conv_bench/amos-conv2d-benchmark/conv2d-benchmark.py --trials 1000 --simple_mode 1 | tee amos_tunning_simple.log

CUDA_VISIBLE_DEVICES=1,2,3 python3 /workspace/v-leiwang3/lowbit-benchmark/2.conv_bench/amos-conv2d-benchmark/conv2d-benchmark.py --trials 1000 --simple_mode 0 | tee amos_tunning_no_simple.log