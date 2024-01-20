import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build, apply_and_build_parallel
import time
from tvm.dlight.base.analysis import get_tensorized_func_and_tags

@T.prim_func(private=True)
def fused_max_pool2d_multiply_add2_relu1(lv3: T.Buffer((T.int64(128), T.int64(112), T.int64(112), T.int64(64)), "float16"), param_0: T.Buffer((T.int64(1), T.int64(1), T.int64(1), T.int64(64)), "float16"), param_1: T.Buffer((T.int64(1), T.int64(1), T.int64(1), T.int64(64)), "float16"), T_relu_intermediate: T.Buffer((T.int64(128), T.int64(56), T.int64(56), T.int64(64)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(128), T.int64(114), T.int64(114), T.int64(64)), "float16")
    pool_max_intermediate = T.alloc_buffer((T.int64(128), T.int64(56), T.int64(56), T.int64(64)), "float16")
    T_multiply_intermediate = T.alloc_buffer((T.int64(128), T.int64(56), T.int64(56), T.int64(64)), "float16")
    T_add_intermediate = T.alloc_buffer((T.int64(128), T.int64(56), T.int64(56), T.int64(64)), "float16")
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(114), T.int64(114), T.int64(64)):
        with T.block("pad_temp"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv3[v_ax0, v_ax1 - T.int64(1), v_ax2 - T.int64(1), v_ax3])
            T.writes(pad_temp[v_ax0, v_ax1, v_ax2, v_ax3])
            pad_temp[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1) <= v_ax1 and v_ax1 < T.int64(113) and T.int64(1) <= v_ax2 and v_ax2 < T.int64(113), lv3[v_ax0, v_ax1 - T.int64(1), v_ax2 - T.int64(1), v_ax3], T.float16(-65504))
    for ax0, ax1, ax2, ax3, rv0, rv1 in T.grid(T.int64(128), T.int64(56), T.int64(56), T.int64(64), T.int64(3), T.int64(3)):
        with T.block("pool_max"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_rv0, v_rv1 = T.axis.remap("SSSSRR", [ax0, ax1, ax2, ax3, rv0, rv1])
            T.reads(pad_temp[v_ax0, v_ax1 * T.int64(2) + v_rv0, v_ax2 * T.int64(2) + v_rv1, v_ax3])
            T.writes(pool_max_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.block_attr({"schedule_rule": "meta_schedule.pool_max"})
            with T.init():
                pool_max_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.float16(-65504)
            pool_max_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(pool_max_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], pad_temp[v_ax0, v_ax1 * T.int64(2) + v_rv0, v_ax2 * T.int64(2) + v_rv1, v_ax3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(56), T.int64(56), T.int64(64)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(pool_max_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], param_0[T.int64(0), T.int64(0), T.int64(0), v_ax3])
            T.writes(T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = pool_max_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * param_0[T.int64(0), T.int64(0), T.int64(0), v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(56), T.int64(56), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], param_1[T.int64(0), T.int64(0), T.int64(0), v_ax3])
            T.writes(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + param_1[T.int64(0), T.int64(0), T.int64(0), v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(56), T.int64(56), T.int64(64)):
        with T.block("T_relu"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_relu_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T_relu_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(0))

func = fused_max_pool2d_multiply_add2_relu1
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)
policy = DefaultPolicy(
    func=func,
    arch=arch,
)
configs = policy.emit_config(20)
for config in configs:
    print(config)

tune_start = time.time()
cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
print(best.latency)
