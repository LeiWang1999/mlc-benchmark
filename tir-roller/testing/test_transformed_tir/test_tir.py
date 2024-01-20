import tvm
from tvm.script import ir as I
from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((1, 1024), "float16"), B: T.Buffer((1024, 1024), "float16"), C: T.Buffer((1, 1024), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    C_rf_local = T.alloc_buffer((128, 1, 1024), "float16", scope="local")
    C_rf_local_1 = T.alloc_buffer((32, 1, 1024), "float16", scope="local")
    B_local = T.alloc_buffer((1024, 1024), "float16", scope="local")
    A_shared = T.alloc_buffer((1, 1024), "float16", scope="shared")
    for u_fused_ax0_fused_fused_0 in T.thread_binding(64, thread="blockIdx.x"):
        for u_fused_ax0_fused_fused_1 in T.thread_binding(16, thread="threadIdx.y"):
            for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 in T.thread_binding(32, thread="threadIdx.x"):
                for ax0 in range(1):
                    for ax1_0 in T.serial(1, annotations={"pragma_unroll_explicit": 256, "pragma_vectorize": 1}):
                        for ax1_1 in T.thread_binding(16, thread="threadIdx.y"):
                            for ax1_2 in T.thread_binding(32, thread="threadIdx.x"):
                                for ax1_3 in T.vectorized(2):
                                    with T.block("A_shared"):
                                        v0 = T.axis.spatial(1, ax0)
                                        v1 = T.axis.spatial(1024, ax1_0 * 1024 + ax1_1 * 64 + ax1_2 * 2 + ax1_3)
                                        T.reads(A[v0, v1])
                                        T.writes(A_shared[v0, v1])
                                        A_shared[v0, v1] = A[v0, v1]
                for u_fused_ax0_fused_fused_2_init in range(1):
                    for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1_init in T.vectorized(4):
                        with T.block("B_rf_init"):
                            vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused = T.axis.spatial(128, ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1_init)
                            v0 = T.axis.spatial(1024, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2_init)
                            T.reads()
                            T.writes(C_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, v0])
                            C_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, v0] = T.float16(0)
                for ax1_fused_u_fused_0 in T.serial(4, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    for ax0_0, ax1 in T.grid(1, 8):
                        for ax0_1 in T.vectorized(1):
                            with T.block("B_local"):
                                v0 = T.axis.spatial(1024, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + ax0_0 + ax0_1)
                                v1 = T.axis.spatial(1024, ax1_fused_u_fused_0 * 256 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 8 + ax1)
                                T.reads(B[v0, v1])
                                T.writes(B_local[v0, v1])
                                B_local[v0, v1] = B[v0, v1]
                    for u_fused_ax0_fused_fused_2, ax1_fused_u_fused_2 in T.grid(1, 2):
                        for ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1 in T.vectorized(4):
                            with T.block("B_rf_update"):
                                vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused = T.axis.spatial(128, ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + ax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1)
                                v0 = T.axis.spatial(1024, u_fused_ax0_fused_fused_0 * 16 + u_fused_ax0_fused_fused_1 + u_fused_ax0_fused_fused_2)
                                vax1_fused_u_fused_0, vax1_fused_u_fused_2 = T.axis.remap("RR", [ax1_fused_u_fused_0, ax1_fused_u_fused_2])
                                T.reads(C_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, v0], A_shared[0, vax1_fused_u_fused_0 * 256 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused // 4 * 8 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused % 4], B_local[v0, vax1_fused_u_fused_0 * 256 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused // 4 * 8 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused % 4])
                                T.writes(C_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, v0])
                                C_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, v0] = C_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused, 0, v0] + A_shared[0, vax1_fused_u_fused_0 * 256 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused // 4 * 8 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused % 4] * B_local[v0, vax1_fused_u_fused_0 * 256 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused // 4 * 8 + vax1_fused_u_fused_2 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused % 4]
        for ax2_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
            for ax0 in T.thread_binding(32, thread="threadIdx.x"):
                for ax2_fused_1_0 in T.serial(1, annotations={"pragma_auto_unroll_max_step": 256, "pragma_unroll_explicit": 1}):
                    for ax2_fused_1_1 in T.vectorized(1):
                        with T.block("B_rf_init"):
                            vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 = T.axis.spatial(32, ax0)
                            v0 = T.axis.spatial(1024, u_fused_ax0_fused_fused_0 * 16 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                            T.reads()
                            T.writes(C_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, v0])
                            C_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, v0] = T.float16(0)
                        for ax1 in range(4):
                            with T.block("B_rf_update"):
                                vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1 = T.axis.remap("SR", [ax0, ax1])
                                v0 = T.axis.spatial(1024, u_fused_ax0_fused_fused_0 * 16 + ax2_fused_0 + ax2_fused_1_0 + ax2_fused_1_1)
                                T.reads(C_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, v0], C_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, 0, v0])
                                T.writes(C_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, v0])
                                C_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, v0] = C_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, v0] + C_rf_local[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 * 4 + vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_1, 0, v0]
        for ax1_fused_1 in range(1):
            for ax1_fused_0 in T.thread_binding(16, thread="threadIdx.y"):
                for ax0 in T.thread_binding(32, thread="threadIdx.x"):
                    with T.block("B"):
                        vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0 = T.axis.reduce(32, ax0)
                        v0 = T.axis.spatial(1024, u_fused_ax0_fused_fused_0 * 16 + ax1_fused_0 + ax1_fused_1)
                        T.reads(C_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, v0])
                        T.writes(C[0, v0])
                        with T.init():
                            C[0, v0] = T.float16(0)
                        C[0, v0] = C[0, v0] + C_rf_local_1[vax1_fused_u_fused_1_ax1_fused_u_fused_3_fused_0, 0, v0]

mod = tvm.IRModule.from_expr(main)
sch = tvm.tir.Schedule(mod, debug_mask="all")
dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
# with open("after_memory_rewrite.cu", "+w") as f:
#     f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())