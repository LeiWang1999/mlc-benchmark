#!/bin/bash
export CUDA_VISIBLE_DEVICES=2
export AMOS_HOME=/workspace/v-leiwang3/AMOS
export PYTHONPATH=$AMOS_HOME/python

ncu --export amos_16384_float16_tensorcore.ncu-rep --force-overwrite --target-processes all --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --section-folder /workspace/v-leiwang3/nsight_compute_workspace/sections --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Nvlink_Tables --section Nvlink_Topology --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start on --cache-control all --clock-control base --apply-rules yes --import-source yes --check-exit-code yes  python3 /workspace/v-leiwang3/lowbit-benchmark/1.matmul_bench/amosgemm/test_gemm_16384.py --trials 0 --simple_mode 1 --in_dtype float16 --out_dtype float16


ncu --export amos_16384_int8_tensorcore.ncu-rep --force-overwrite --target-processes all --replay-mode kernel --kernel-name-base function --launch-skip-before-match 0 --section-folder /workspace/v-leiwang3/nsight_compute_workspace/sections --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Nvlink_Tables --section Nvlink_Topology --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start on --cache-control all --clock-control base --apply-rules yes --import-source yes --check-exit-code yes  python3 /workspace/v-leiwang3/lowbit-benchmark/1.matmul_bench/amosgemm/test_gemm_16384.py --trials 0 --simple_mode 1 --in_dtype int8 --out_dtype int32
