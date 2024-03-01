import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import get_mma_intrin_group

@T.prim_func
def fused_fused_decode5_fused_NT_matmul10_add1(lv55: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv56: T.Buffer((T.int64(4096), T.int64(344)), "float16"), lv54: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv50: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), T_add_intermediate_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    NT_matmul_intermediate_local = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
    NT_matmul_intermediate_rf_local = T.alloc_buffer((T.int64(128), T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
    NT_matmul_intermediate_rf_local_1 = T.alloc_buffer((T.int64(32), T.int64(1), T.int64(1), T.int64(4096)), "float16", scope="local")
    lv55_local = T.alloc_buffer((T.int64(4096), T.int64(1376)), "uint32", scope="local")
    lv54_shared = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16", scope="shared")
    for u_fused_ax0_fused_fused_0 in T.thread_binding(T.int64(256), thread="blockIdx.x"):
        for u_fused_ax0_fused_fused_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
            for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for ax0, ax1 in T.grid(T.int64(1), T.int64(1)):
                    for ax2_0 in T.serial(T.int64(22), annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                        for ax2_1 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                            for ax2_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                for ax2_3 in T.vectorized(T.int64(1)):
                                    with T.block("lv54_shared"):
                                        v0, v1 = T.axis.remap("SS", [ax0, ax1])
                                        v2 = T.axis.spatial(T.int64(11008), ax2_0 * T.int64(512) + ax2_1 * T.int64(32) + ax2_2 + ax2_3)
                                        T.where((ax2_0 * T.int64(16) + ax2_1) * T.int64(32) + ax2_2 + ax2_3 < T.int64(11008))
                                        T.reads(lv54[v0, v1, v2])
                                        T.writes(lv54_shared[v0, v1, v2])
                                        lv54_shared[v0, v1, v2] = lv54[v0, v1, v2]
                for u_fused_ax0_fused_fused_2_init in range(T.int64(1)):
                    for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init in T.vectorized(T.int64(4)):
                        with T.block("NT_matmul_rf_init"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(T.int64(128), ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1_init)
                            v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                            T.reads()
                            T.writes(NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0])
                            NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] = T.float16(0)
                for ax1_0_fused_ax1_1_fused_0 in T.serial(T.int64(43), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    for ax0_0, ax1 in T.grid(T.int64(1), T.int64(1)):
                        for ax0_1 in T.vectorized(T.int64(1)):
                            with T.block("lv55_local"):
                                v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1 + ax0_0 + ax0_1)
                                v1 = T.axis.spatial(T.int64(1376), ax1_0_fused_ax1_1_fused_0 * T.int64(32) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 + ax1)
                                T.reads(lv55[v0, v1])
                                T.writes(lv55_local[v0, v1])
                                lv55_local[v0, v1] = lv55[v0, v1]
                    for u_fused_ax0_fused_fused_2, ax1_0_fused_ax1_1_fused_2 in T.grid(T.int64(1), T.int64(2)):
                        for ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 in T.vectorized(T.int64(4)):
                            with T.block("NT_matmul_rf_update"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused = T.axis.spatial(T.int64(128), ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + ax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1)
                                v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                vax1_0_fused_ax1_1_fused_0, vax1_0_fused_ax1_1_fused_2 = T.axis.remap("RR", [ax1_0_fused_ax1_1_fused_0, ax1_0_fused_ax1_1_fused_2])
                                T.reads(NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0], lv54_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)], lv55_local[v0, vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) + vax1_0_fused_ax1_1_fused_2 // T.int64(2)], lv56[v0, (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) // T.int64(32)])
                                T.writes(NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0])
                                NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] = NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused, T.int64(0), T.int64(0), v0] + lv54_shared[T.int64(0), T.int64(0), vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)] * ((T.Cast("float16", T.bitwise_and(T.shift_right(lv55_local[v0, vax1_0_fused_ax1_1_fused_0 * T.int64(32) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) + vax1_0_fused_ax1_1_fused_2 // T.int64(2)], T.Cast("uint32", (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv56[v0, (vax1_0_fused_ax1_1_fused_0 * T.int64(256) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused // T.int64(4) * T.int64(8) + vax1_0_fused_ax1_1_fused_2 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused % T.int64(4)) // T.int64(32)])
        for ax2_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
            for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                for ax2_fused_1_0 in T.serial(T.int64(1), annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    for ax2_fused_1_1 in T.vectorized(T.int64(1)):
                        with T.block("NT_matmul_rf_init"):
                            vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.spatial(T.int64(32), ax0)
                            v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                            T.reads()
                            T.writes(NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                            NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] = T.float16(0)
                        for ax1 in range(T.int64(4)):
                            with T.block("NT_matmul_rf_update"):
                                vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                T.reads(NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0], NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, T.int64(0), T.int64(0), v0])
                                T.writes(NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                                NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] = NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0] + NT_matmul_intermediate_rf_local[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 * T.int64(4) + vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_1, T.int64(0), T.int64(0), v0]
        for ax1_fused_1 in range(T.int64(1)):
            for ax1_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
                for ax0 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                    with T.block("NT_matmul"):
                        vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0 = T.axis.reduce(T.int64(32), ax0)
                        v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax1_fused_0 + ax1_fused_1)
                        T.reads(NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0])
                        T.writes(NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                        with T.init():
                            NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] = T.float16(0)
                        NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] = NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0] + NT_matmul_intermediate_rf_local_1[vax1_0_fused_ax1_1_fused_1_ax1_0_fused_ax1_1_fused_3_fused_0, T.int64(0), T.int64(0), v0]
        for ax0_fused_0 in T.thread_binding(T.int64(16), thread="threadIdx.y"):
            for ax0_fused_1 in range(T.int64(1)):
                with T.block("T_add"):
                    v0 = T.axis.spatial(T.int64(4096), u_fused_ax0_fused_fused_0 * T.int64(16) + ax0_fused_0 + ax0_fused_1)
                    T.reads(lv50[T.int64(0), T.int64(0), v0], NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0])
                    T.writes(T_add_intermediate_intermediate[T.int64(0), T.int64(0), v0])
                    T_add_intermediate_intermediate[T.int64(0), T.int64(0), v0] = lv50[T.int64(0), T.int64(0), v0] + NT_matmul_intermediate_local[T.int64(0), T.int64(0), v0]

mod = tvm.IRModule.from_expr(fused_fused_decode5_fused_NT_matmul10_add1)
dense_relu_0_rt_mod = tvm.build(mod, target="cuda")
with open("after_memory_rewrite.cu", "+w") as f:
    f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())
    