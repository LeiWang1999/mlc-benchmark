import os.path as osp
import tempfile
from typing import Callable, List, Optional

import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm import relay, tir
from tvm.ir.module import IRModule
from tvm.meta_schedule.database import TuningRecord, Workload
from tvm.script import tir as T
from tvm.target import Target
from tvm.tir import Schedule
import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build, apply_and_build_parallel
import time

def _create_tmp_database(tmpdir: str, mod_eq: str = "structural") -> ms.database.JSONDatabase:
    path_workload = osp.join(tmpdir, "workloads.json")
    path_tuning_record = osp.join(tmpdir, "tuning_records.json")
    return ms.database.JSONDatabase(path_workload, path_tuning_record, module_equality=mod_eq)


def matmul_nt(M, N, K, in_dtype="float16", out_dtype="float16"):
    @tvm.script.ir_module
    class MatmulNT:
        @T.prim_func
        def main(a: T.handle, b: T.handle, c: T.handle):
            T.func_attr({"global_symbol": "main", "tir.noalias": True})
            A = T.match_buffer(a, [M, K], dtype=in_dtype)
            B = T.match_buffer(b, [N, K], dtype=in_dtype)
            C = T.match_buffer(c, [M, N], dtype=out_dtype)

            for i, j, k in T.grid(M, N, K):
                with T.block("B"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = 0.0
                    C[vi, vj] = C[vi, vj] + A[vi, vk].astype(out_dtype) * B[vj, vk].astype(
                        out_dtype
                    )

    return MatmulNT


benchmark_sets = [
    # (prim_func, input_args, fast_dlight_schedule, default_dlight_schedule),
    (matmul_nt, (1024, 1024, 1024, "float16", "float16"), Matmul, Matmul),
]
benchmark_results = {}
for get_prim_func, input_args, f_schedule, d_schedule in benchmark_sets:
    ir_module = get_prim_func(*input_args)
    func = ir_module["main"]
    target = tvm.target.Target("nvidia/nvidia-a100")
    arch = CUDA(target)
    policy = TensorCorePolicy(
        func=func,
        arch=arch,
        tags={
            "tensorcore_config": [0, 1],
            "pipeline_stage": 2,
            "use_async_copy": 1,
        },
    )
    configs = policy.emit_config(1)
    for config in configs:
        print(config)

    tune_start = time.time()
    cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
    print(best.sch.mod)
    print(best.code)
    fast_tune_time = time.time() - tune_start
    print("[FastDlight] The best latency of top 1 is {:.3f} ms".format(cpresults[0].latency))
    print("[FastDlight] The best latency of top 20 is {:.3f} ms".format(best.latency))

    # evaluate the performance of the default schedule

    rule = d_schedule()
    default_tune_start = time.time()
    sch_default = rule.apply(func, target, False)
    with tvm.transform.PassContext(config={"tir.use_async_copy": True}):
        mod_default = tvm.build(sch_default.mod["main"], target="cuda")
    default_tune_time = time.time() - default_tune_start

    args = func.buffer_map.values()

    profile_tensors = []
    for arg in args:
        profile_tensors.append(
            tvm.nd.array(
                np.random.uniform(0, 1, [int(i) for i in arg.shape]).astype(arg.dtype),
                device=arch.device,
            )
        )

    timer_cuda_mod = mod_default.time_evaluator(mod_default.entry_name, arch.device, number=5)
    t = timer_cuda_mod(*profile_tensors).mean

    print("Time cost of Dlight default schedule: {:.3f} ms".format(t * 1e3))

    profile_config = {
        f"{get_prim_func.__name__}-{'-'.join([str(i) for i in input_args])}": {
            "fast_dlight_top20_tune_time": fast_tune_time,
            "fast_dlight_top1_latency": cpresults[0].latency,
            "fast_dlight_top20_latency": best.latency,
            "default_dlight_tune_time": default_tune_time,
            "default_dlight_latency": t * 1e3,
        }
    }
    benchmark_results.update(profile_config)


    with tempfile.TemporaryDirectory() as tmpdir:
        database = _create_tmp_database(tmpdir)
        workload = database.commit_workload(ir_module)
        record = ms.database.TuningRecord(
            best.sch.trace,
            workload,
            [1.5, 2.5, 1.8], # runtime (actual runtime instead of tuning time.)
            arch.target,
            ms.arg_info.ArgInfo.from_prim_func(func=best.sch.mod["main"]),
        )
        database.commit_tuning_record(record)
        new_database = ms.database.JSONDatabase(
            path_workload=database.path_workload,
            path_tuning_record=database.path_tuning_record,
        )
        
