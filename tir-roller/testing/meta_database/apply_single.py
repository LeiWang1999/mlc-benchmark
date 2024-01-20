import tvm
from tvm.script import ir as I
from tvm.script import tir as T
import os.path as osp
from tvm.tir.tensor_intrin.cuda import get_wmma_intrin_group

@T.prim_func
def fused_dense_add_relu(lv: T.Buffer((T.int64(128), T.int64(150528)), "float16"), param_0: T.Buffer((T.int64(128), T.int64(150528)), "float16"), param_1: T.Buffer((T.int64(1), T.int64(128)), "float16"), var_T_relu_intermediate: T.Buffer((T.int64(128), T.int64(128)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_matmul_NT_intermediate = T.alloc_buffer((T.int64(128), T.int64(128)), "float16")
    var_T_add_intermediate = T.alloc_buffer((T.int64(128), T.int64(128)), "float16")
    for i0, i1, k in T.grid(T.int64(128), T.int64(128), T.int64(150528)):
        with T.block("T_matmul_NT"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv[v_i0, v_k], param_0[v_i1, v_k])
            T.writes(var_T_matmul_NT_intermediate[v_i0, v_i1])
            with T.init():
                var_T_matmul_NT_intermediate[v_i0, v_i1] = T.float16(0)
            var_T_matmul_NT_intermediate[v_i0, v_i1] = var_T_matmul_NT_intermediate[v_i0, v_i1] + lv[v_i0, v_k] * param_0[v_i1, v_k]
    for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_matmul_NT_intermediate[v_ax0, v_ax1], param_1[T.int64(0), v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_T_matmul_NT_intermediate[v_ax0, v_ax1] + param_1[T.int64(0), v_ax1]
    for ax0, ax1 in T.grid(T.int64(128), T.int64(128)):
        with T.block("T_relu"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1])
            T.writes(var_T_relu_intermediate[v_ax0, v_ax1])
            var_T_relu_intermediate[v_ax0, v_ax1] = T.max(var_T_add_intermediate[v_ax0, v_ax1], T.float16(0))

from tvm import meta_schedule as ms

temp_dir = "./e2e_mlp"
path_workload = osp.join(temp_dir, "database_workload.json")
path_tuning_record = osp.join(temp_dir, "database_tuning_record.json")
cache_meta_database = ms.database.JSONDatabase(
    path_workload, path_tuning_record, module_equality="structural"
)
normalize_mod_func_ = tvm._ffi.get_global_func("tvm.meta_schedule.normalize_mod");
ir_mod = normalize_mod_func_(fused_dense_add_relu)
cache_meta_database