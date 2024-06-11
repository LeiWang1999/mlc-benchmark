export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$(pwd)/tvm/python
python -u ./cutlass_fpa_intb.py 2>&1 | tee cutlass_fpa_intb.log
# python -u ./cutlass_fpa_int8.py 2>&1 | tee cutlass_fpa_int8.log
