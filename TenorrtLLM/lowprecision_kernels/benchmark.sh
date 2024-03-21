export CUDA_VISIBLE_DEVICES=1

python -u ./cutlass_fpa_intb.py 2>&1 | tee cutlass_fpa_intb.log
