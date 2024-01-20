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
def fused_conv2d10_add9_multiply3_add8_relu4(
    lv70: T.Buffer((T.int64(128), T.int64(7), T.int64(7), T.int64(512)), "float16"),
    param_0: T.Buffer((T.int64(3), T.int64(3), T.int64(512), T.int64(512)), "float16"),
    lv64: T.Buffer((T.int64(128), T.int64(7), T.int64(7), T.int64(512)), "float16"),
    param_1: T.Buffer((T.int64(1), T.int64(1), T.int64(1), T.int64(512)), "float16"),
    param_2: T.Buffer((T.int64(1), T.int64(1), T.int64(1), T.int64(512)), "float16"),
    var_T_relu_intermediate: T.Buffer(
        (T.int64(128), T.int64(7), T.int64(7), T.int64(512)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(128), T.int64(9), T.int64(9), T.int64(512)), "float16")
    var_conv2d_nhwc_intermediate = T.alloc_buffer(
        (T.int64(128), T.int64(7), T.int64(7), T.int64(512)), "float16"
    )
    var_T_add_intermediate = T.alloc_buffer(
        (T.int64(128), T.int64(7), T.int64(7), T.int64(512)), "float16"
    )
    var_T_multiply_intermediate = T.alloc_buffer(
        (T.int64(128), T.int64(7), T.int64(7), T.int64(512)), "float16"
    )
    var_T_add_intermediate_1 = T.alloc_buffer(
        (T.int64(128), T.int64(7), T.int64(7), T.int64(512)), "float16"
    )
    for i0, i1, i2, i3 in T.grid(T.int64(128), T.int64(9), T.int64(9), T.int64(512)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv70[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(
                T.int64(1) <= v_i1
                and v_i1 < T.int64(8)
                and T.int64(1) <= v_i2
                and v_i2 < T.int64(8),
                lv70[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3],
                T.float16(0),
            )
    for nn, yy, xx, ff, ry, rx, rc in T.grid(
        T.int64(128), T.int64(7), T.int64(7), T.int64(512), T.int64(3), T.int64(3), T.int64(512)
    ):
        with T.block("conv2d_nhwc"):
            v_nn, v_yy, v_xx, v_ff, v_ry, v_rx, v_rc = T.axis.remap(
                "SSSSRRR", [nn, yy, xx, ff, ry, rx, rc]
            )
            T.reads(pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc], param_0[v_ry, v_rx, v_rc, v_ff])
            T.writes(var_conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff])
            with T.init():
                var_conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff] = T.float16(0)
            var_conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff] = (
                var_conv2d_nhwc_intermediate[v_nn, v_yy, v_xx, v_ff]
                + pad_temp[v_nn, v_yy + v_ry, v_xx + v_rx, v_rc] * param_0[v_ry, v_rx, v_rc, v_ff]
            )
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(7), T.int64(7), T.int64(512)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                var_conv2d_nhwc_intermediate[v_ax0, v_ax1, v_ax2, v_ax3],
                lv64[v_ax0, v_ax1, v_ax2, v_ax3],
            )
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = (
                var_conv2d_nhwc_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
                + lv64[v_ax0, v_ax1, v_ax2, v_ax3]
            )
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(7), T.int64(7), T.int64(512)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3],
                param_1[T.int64(0), T.int64(0), T.int64(0), v_ax3],
            )
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = (
                var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
                * param_1[T.int64(0), T.int64(0), T.int64(0), v_ax3]
            )
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(7), T.int64(7), T.int64(512)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3],
                param_2[T.int64(0), T.int64(0), T.int64(0), v_ax3],
            )
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = (
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
                + param_2[T.int64(0), T.int64(0), T.int64(0), v_ax3]
            )
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(7), T.int64(7), T.int64(512)):
        with T.block("T_relu"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_relu_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_relu_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(
                var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(0)
            )


func = fused_conv2d10_add9_multiply3_add8_relu4
# func = matmul
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)
_tensorized_func, tags = get_tensorized_func_and_tags(func, arch.target)
policy = TensorCorePolicy(
    func=_tensorized_func,
    arch=arch,
    tags=tags,
)
configs = policy.emit_config(1)
for config in configs:
    print(config)

tune_start = time.time()
cpresults, best = apply_and_build(func, configs, arch, parallel_build=False)
print(best.latency)
