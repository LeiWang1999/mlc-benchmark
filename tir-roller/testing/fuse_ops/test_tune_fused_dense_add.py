import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build, apply_and_build_parallel
import time


@T.prim_func
def fused_dense_add(
    lv77: T.Buffer((T.int64(128), T.int64(512)), "float16"),
    param_0: T.Buffer((T.int64(1000), T.int64(512)), "float16"),
    param_1: T.Buffer((T.int64(1), T.int64(1000)), "float16"),
    var_T_add_intermediate: T.Buffer((T.int64(128), T.int64(1000)), "float16"),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_matmul_NT_intermediate = T.alloc_buffer((T.int64(128), T.int64(1000)), "float16")
    for i0, i1, k in T.grid(T.int64(128), T.int64(1000), T.int64(512)):
        with T.block("T_matmul_NT"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv77[v_i0, v_k], param_0[v_i1, v_k])
            T.writes(var_T_matmul_NT_intermediate[v_i0, v_i1])
            with T.init():
                var_T_matmul_NT_intermediate[v_i0, v_i1] = T.float16(0)
            var_T_matmul_NT_intermediate[v_i0, v_i1] = (
                var_T_matmul_NT_intermediate[v_i0, v_i1] + lv77[v_i0, v_k] * param_0[v_i1, v_k]
            )
    for ax0, ax1 in T.grid(T.int64(128), T.int64(1000)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_matmul_NT_intermediate[v_ax0, v_ax1], param_1[T.int64(0), v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = (
                var_T_matmul_NT_intermediate[v_ax0, v_ax1] + param_1[T.int64(0), v_ax1]
            )


func = fused_dense_add
# func = matmul
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
configs = policy.emit_config(20)
for config in configs:
    print(config)

tune_start = time.time()
cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
print(best.latency)
