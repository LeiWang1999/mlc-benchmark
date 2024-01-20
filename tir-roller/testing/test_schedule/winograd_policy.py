import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build, apply_and_build_parallel
import time


@T.prim_func
def fused_contrib_conv2d_winograd_without_weight_transform1_add5_add4_relu2(lv32: T.Buffer((T.int64(128), T.int64(28), T.int64(28), T.int64(128)), "float32"), param_0: T.Buffer((T.int64(4), T.int64(4), T.int64(128), T.int64(128)), "float32"), lv27: T.Buffer((T.int64(128), T.int64(28), T.int64(28), T.int64(128)), "float32"), param_1: T.Buffer((T.int64(1), T.int64(1), T.int64(1), T.int64(128)), "float32"), var_T_relu_intermediate: T.Buffer((T.int64(128), T.int64(28), T.int64(28), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    data_pad = T.alloc_buffer((T.int64(128), T.int64(30), T.int64(30), T.int64(128)))
    input_tile = T.alloc_buffer((T.int64(4), T.int64(4), T.int64(25088), T.int64(128)))
    B = T.alloc_buffer((T.int64(4), T.int64(4)))
    data_pack = T.alloc_buffer((T.int64(4), T.int64(4), T.int64(25088), T.int64(128)))
    bgemm = T.alloc_buffer((T.int64(4), T.int64(4), T.int64(25088), T.int64(128)))
    A = T.alloc_buffer((T.int64(4), T.int64(2)))
    inverse = T.alloc_buffer((T.int64(2), T.int64(2), T.int64(25088), T.int64(128)))
    var_conv2d_winograd_intermediate = T.alloc_buffer((T.int64(128), T.int64(28), T.int64(28), T.int64(128)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(128), T.int64(28), T.int64(28), T.int64(128)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(128), T.int64(28), T.int64(28), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(128), T.int64(30), T.int64(30), T.int64(128)):
        with T.block("data_pad"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv32[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3])
            T.writes(data_pad[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"schedule_rule": "None"})
            data_pad[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i1 and v_i1 < T.int64(29) and T.int64(1) <= v_i2 and v_i2 < T.int64(29), lv32[v_i0, v_i1 - T.int64(1), v_i2 - T.int64(1), v_i3], T.float32(0))
    for eps, nu, p, ci in T.grid(T.int64(4), T.int64(4), T.int64(25088), T.int64(128)):
        with T.block("input_tile"):
            v_eps, v_nu, v_p, v_ci = T.axis.remap("SSSS", [eps, nu, p, ci])
            T.reads(data_pad[v_p // T.int64(196), v_p % T.int64(196) // T.int64(14) * T.int64(2) + v_eps, v_p % T.int64(14) * T.int64(2) + v_nu, v_ci])
            T.writes(input_tile[v_eps, v_nu, v_p, v_ci])
            T.block_attr({"schedule_rule": "None"})
            input_tile[v_eps, v_nu, v_p, v_ci] = data_pad[v_p // T.int64(196), v_p % T.int64(196) // T.int64(14) * T.int64(2) + v_eps, v_p % T.int64(14) * T.int64(2) + v_nu, v_ci]
    for i, j in T.grid(T.int64(4), T.int64(4)):
        with T.block("B"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads()
            T.writes(B[v_i, v_j])
            T.block_attr({"schedule_rule": "None"})
            B[v_i, v_j] = T.Select(v_i % T.int64(4) == T.int64(3) and v_j % T.int64(4) == T.int64(3), T.float32(1), T.Select(v_i % T.int64(4) == T.int64(3) and v_j % T.int64(4) == T.int64(2), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(3) and v_j % T.int64(4) == T.int64(1), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(3) and v_j % T.int64(4) == T.int64(0), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(2) and v_j % T.int64(4) == T.int64(3), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(2) and v_j % T.int64(4) == T.int64(2), T.float32(1), T.Select(v_i % T.int64(4) == T.int64(2) and v_j % T.int64(4) == T.int64(1), T.float32(1), T.Select(v_i % T.int64(4) == T.int64(2) and v_j % T.int64(4) == T.int64(0), T.float32(-1), T.Select(v_i % T.int64(4) == T.int64(1) and v_j % T.int64(4) == T.int64(3), T.float32(-1), T.Select(v_i % T.int64(4) == T.int64(1) and v_j % T.int64(4) == T.int64(2), T.float32(1), T.Select(v_i % T.int64(4) == T.int64(1) and v_j % T.int64(4) == T.int64(1), T.float32(-1), T.Select(v_i % T.int64(4) == T.int64(1) and v_j % T.int64(4) == T.int64(0), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(0) and v_j % T.int64(4) == T.int64(3), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(0) and v_j % T.int64(4) == T.int64(2), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(0) and v_j % T.int64(4) == T.int64(1), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(0) and v_j % T.int64(4) == T.int64(0), T.float32(1), T.float32(0)))))))))))))))))
    for eps, nu, p, ci, r_a, r_b in T.grid(T.int64(4), T.int64(4), T.int64(25088), T.int64(128), T.int64(4), T.int64(4)):
        with T.block("data_pack"):
            v_eps, v_nu, v_p, v_ci, v_r_a, v_r_b = T.axis.remap("SSSSRR", [eps, nu, p, ci, r_a, r_b])
            T.reads(input_tile[v_r_a, v_r_b, v_p, v_ci], B[T.min(v_r_a, v_r_b):T.min(v_r_a, v_r_b) + (T.max(v_r_a, v_r_b) + T.int64(1) - T.min(v_r_a, v_r_b)), T.min(v_eps, v_nu):T.min(v_eps, v_nu) + (T.max(v_eps, v_nu) + T.int64(1) - T.min(v_eps, v_nu))])
            T.writes(data_pack[v_eps, v_nu, v_p, v_ci])
            T.block_attr({"schedule_rule": "conv2d_nhwc_winograd_data_pack"})
            with T.init():
                data_pack[v_eps, v_nu, v_p, v_ci] = T.float32(0)
            data_pack[v_eps, v_nu, v_p, v_ci] = data_pack[v_eps, v_nu, v_p, v_ci] + input_tile[v_r_a, v_r_b, v_p, v_ci] * B[v_r_a, v_eps] * B[v_r_b, v_nu]
    for eps, nu, p, co, ci in T.grid(T.int64(4), T.int64(4), T.int64(25088), T.int64(128), T.int64(128)):
        with T.block("bgemm"):
            v_eps, v_nu, v_p, v_co, v_ci = T.axis.remap("SSSSR", [eps, nu, p, co, ci])
            T.reads(data_pack[v_eps, v_nu, v_p, v_ci], param_0[v_eps, v_nu, v_co, v_ci])
            T.writes(bgemm[v_eps, v_nu, v_p, v_co])
            with T.init():
                bgemm[v_eps, v_nu, v_p, v_co] = T.float32(0)
            bgemm[v_eps, v_nu, v_p, v_co] = bgemm[v_eps, v_nu, v_p, v_co] + data_pack[v_eps, v_nu, v_p, v_ci] * param_0[v_eps, v_nu, v_co, v_ci]
    for i, j in T.grid(T.int64(4), T.int64(2)):
        with T.block("A"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads()
            T.writes(A[v_i, v_j])
            T.block_attr({"schedule_rule": "None"})
            A[v_i, v_j] = T.Select(v_i % T.int64(4) == T.int64(3) and v_j % T.int64(2) == T.int64(1), T.float32(1), T.Select(v_i % T.int64(4) == T.int64(3) and v_j % T.int64(2) == T.int64(0), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(2) and v_j % T.int64(2) == T.int64(1), T.float32(1), T.Select(v_i % T.int64(4) == T.int64(2) and v_j % T.int64(2) == T.int64(0), T.float32(1), T.Select(v_i % T.int64(4) == T.int64(1) and v_j % T.int64(2) == T.int64(1), T.float32(-1), T.Select(v_i % T.int64(4) == T.int64(1) and v_j % T.int64(2) == T.int64(0), T.float32(1), T.Select(v_i % T.int64(4) == T.int64(0) and v_j % T.int64(2) == T.int64(1), T.float32(0), T.Select(v_i % T.int64(4) == T.int64(0) and v_j % T.int64(2) == T.int64(0), T.float32(1), T.float32(0)))))))))
    for vh, vw, p, co, r_a, r_b in T.grid(T.int64(2), T.int64(2), T.int64(25088), T.int64(128), T.int64(4), T.int64(4)):
        with T.block("inverse"):
            v_vh, v_vw, v_p, v_co, v_r_a, v_r_b = T.axis.remap("SSSSRR", [vh, vw, p, co, r_a, r_b])
            T.reads(bgemm[v_r_a, v_r_b, v_p, v_co], A[T.min(v_r_a, v_r_b):T.min(v_r_a, v_r_b) + (T.max(v_r_a, v_r_b) + T.int64(1) - T.min(v_r_a, v_r_b)), T.min(v_vh, v_vw):T.min(v_vh, v_vw) + (T.max(v_vh, v_vw) + T.int64(1) - T.min(v_vh, v_vw))])
            T.writes(inverse[v_vh, v_vw, v_p, v_co])
            T.block_attr({"schedule_rule": "conv2d_nhwc_winograd_inverse"})
            with T.init():
                inverse[v_vh, v_vw, v_p, v_co] = T.float32(0)
            inverse[v_vh, v_vw, v_p, v_co] = inverse[v_vh, v_vw, v_p, v_co] + bgemm[v_r_a, v_r_b, v_p, v_co] * A[v_r_a, v_vh] * A[v_r_b, v_vw]
    for n, h, w, co in T.grid(T.int64(128), T.int64(28), T.int64(28), T.int64(128)):
        with T.block("conv2d_winograd"):
            v_n, v_h, v_w, v_co = T.axis.remap("SSSS", [n, h, w, co])
            T.reads(inverse[v_h % T.int64(2), v_w % T.int64(2), v_n * T.int64(196) + v_h // T.int64(2) * T.int64(14) + v_w // T.int64(2), v_co])
            T.writes(var_conv2d_winograd_intermediate[v_n, v_h, v_w, v_co])
            var_conv2d_winograd_intermediate[v_n, v_h, v_w, v_co] = inverse[v_h % T.int64(2), v_w % T.int64(2), v_n * T.int64(196) + v_h // T.int64(2) * T.int64(14) + v_w // T.int64(2), v_co]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(28), T.int64(28), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_winograd_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv27[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_winograd_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv27[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(28), T.int64(28), T.int64(128)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], param_1[T.int64(0), T.int64(0), T.int64(0), v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + param_1[T.int64(0), T.int64(0), T.int64(0), v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(128), T.int64(28), T.int64(28), T.int64(128)):
        with T.block("T_relu"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_relu_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_relu_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], T.float32(0))


func = fused_contrib_conv2d_winograd_without_weight_transform1_add5_add4_relu2
target = tvm.target.Target("nvidia/nvidia-a100")
arch = CUDA(target)
policy = DefaultPolicy(func=func, arch=arch)
configs = policy.emit_config(10)
cpresults, best = apply_and_build(func, configs, arch, parallel_build=True)

