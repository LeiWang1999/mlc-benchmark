python matmul.py --benchmark_sets llm_shape_fp16xfp16

python matmul.py --benchmark_sets llm_shape_int8xint8

python matmul_dequantize_int1.py --benchmark_sets llm_int8xint1

python matmul_dequantize_int1.py --batch_seq --benchmark_sets llm_int8xint1

python matmul_dequantize_int4.py --group_size -1 --benchmark_sets llm_shape_fp16xint4

python matmul_dequantize_af.py --group_size 128 --benchmark_sets llm_shape_fp16xnf4

python matmul_dequantize_fp.py --group_size 128 --benchmark_sets llm_shape_fp16xfp4
