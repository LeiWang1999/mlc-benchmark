python matmul.py --benchmark_sets llm_shape_fp16xfp16

python matmul.py --benchmark_sets llm_shape_int8xint8

python matmul_dequantize_int4.py --group_size -1 --benchmark_sets llm_shape_fp16xint4

python matmul_dequantize_af.py --group_size 128 --benchmark_sets llm_shape_fp16xnf4
