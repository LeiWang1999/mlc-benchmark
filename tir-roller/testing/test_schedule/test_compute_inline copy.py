import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import get_mma_intrin_group
# from tvm.script import tir as T
# from tvm.script import ir as I
# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((16384, 16384), "float16"), B: T.Buffer((1024, 1024, 16, 8), "int8"), C: T.Buffer((16384, 16384), "float16")):
    T.func_attr({"dequantize_info": {"B": {"decode_block": "B_decode", "fast_decoding": T.bool(True), "source_format": {"bits": 4, "format": "int"}, "target_format": {"bits": 16, "format": "float"}}}, "dlight.tensorcore_prenormlized": T.bool(True), "smooth_b": T.bool(True), "tir.noalias": T.bool(True)})
    # with T.block("root"):
    B_decode = T.alloc_buffer((1024, 1024, 16, 16), "float16")
    B_reindex = T.alloc_buffer((16384, 16384), "float16")
    A_reindex = T.alloc_buffer((1, 16384, 16384), "float16")
    B_reindex_reindex = T.alloc_buffer((1, 1024, 1024, 16, 16), "float16")
    C_reindex = T.alloc_buffer((1, 16384, 16384), "float16")
    A_reindex_shared_dyn = T.alloc_buffer((1, 16384, 16384), "float16", scope="shared.dyn")
    B_reindex_reindex_shared_dyn = T.alloc_buffer((1, 1024, 1024, 16, 16), "float16", scope="shared.dyn")
    for n, k, nn, kk in T.grid(1024, 1024, 16, 16):
        with T.block("B_decode"):
            v_n, v_k, v_nn, v_kk = T.axis.remap("SSSS", [n, k, nn, kk])
            T.reads(B[v_n, v_k, v_nn, v_kk // 2])
            T.writes(B_decode[v_n, v_k, v_nn, v_kk])
            B_decode[v_n, v_k, v_nn, v_kk] = T.Cast("float16", T.bitwise_and(T.shift_right(B[v_n, v_k, v_nn, v_kk // 2], T.Cast("int8", v_k % 2 * 4)), T.int8(15)))
    for i, j in T.grid(16384, 16384):
        with T.block("B_reindex"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(B_decode[v_i // 16, v_j // 16, v_i % 16, v_j % 16])
            T.writes(B_reindex[v_i, v_j])
            B_reindex[v_i, v_j] = B_decode[v_i // 16, v_j // 16, v_i % 16, v_j % 16]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("A_reindex_reindex"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v0, v1])
            T.writes(A_reindex[0, v0, v1])
            A_reindex[0, v0, v1] = A[v0, v1]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("B_reindex_reindex_reindex"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B_reindex[v0, v1])
            T.writes(B_reindex_reindex[0, v0 // 16, v1 // 16, v0 % 16, v1 % 16])
            B_reindex_reindex[0, v0 // 16, v1 // 16, v0 % 16, v1 % 16] = B_reindex[v0, v1]
    for ax0 in range(1):
        for ax1_0_0_ax2_0_0_fused in T.thread_binding(128, thread="blockIdx.y"):
            for ax1_0_1_ax2_0_1_fused in T.thread_binding(256, thread="blockIdx.x"):
                for ax2_0_2 in T.thread_binding(2, thread="threadIdx.z"):
                    for ax1_0_2 in T.thread_binding(2, thread="threadIdx.y"):
                        for ax3_0_0 in range(512):
                            for ax0_ax1_ax2_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                for ax0_ax1_ax2_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                    for ax0_ax1_ax2_fused_2 in range(4):
                                        for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                with T.block("A_reindex_shared.dyn"):
                                                    v0 = T.axis.spatial(1, 0)
                                                    v1 = T.axis.spatial(16384, ax1_0_0_ax2_0_0_fused * 128 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 1024 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 32)
                                                    v2 = T.axis.spatial(16384, ax3_0_0 * 32 + (ax0_ax1_ax2_fused_0 * 2048 + ax0_ax1_ax2_fused_1 * 1024 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 32)
                                                    T.reads(A_reindex[v0, v1, v2])
                                                    T.writes(A_reindex_shared_dyn[v0, v1, v2])
                                                    A_reindex_shared_dyn[v0, v1, v2] = A_reindex[v0, v1, v2]
                            for ax0_1, ax1, ax2, ax3 in T.grid(4, 2, 16, 16):
                                with T.block("B_reindex_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + ax0_1)
                                    v2 = T.axis.spatial(1024, ax3_0_0 * 2 + ax1)
                                    v3, v4 = T.axis.remap("SS", [ax2, ax3])
                                    T.reads(B_reindex_reindex[v0, v1, v2, v3, v4])
                                    T.writes(B_reindex_reindex_shared_dyn[v0, v1, v2, v3, v4])
                                    B_reindex_reindex_shared_dyn[v0, v1, v2, v3, v4] = B_reindex_reindex[v0, v1, v2, v3, v4]
                            for ax3_0_1, ax1_0_3, ax2_0_3 in T.grid(2, 4, 2):
                                with T.block("C_o"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial(1024, ax1_0_0_ax2_0_0_fused * 8 + ax1_0_2 * 4 + ax1_0_3)
                                    v2_o = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax2_0_3)
                                    v3_o = T.axis.reduce(1024, ax3_0_0 * 2 + ax3_0_1)
                                    T.reads(A_reindex_shared_dyn[0, v1_o * 16:v1_o * 16 + 16, v3_o * 16:v3_o * 16 + 16], B_reindex_reindex_shared_dyn[0, v2_o, v3_o, 0:16, 0:16])
                                    T.writes(C_reindex[0, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                    with T.init():
                                        for ax1_1, ax2_1 in T.grid(16, 16):
                                            with T.block("C_init"):
                                                v1_i_init, v2_i_init = T.axis.remap("SS", [ax1_1, ax2_1])
                                                T.reads()
                                                T.writes(C_reindex[0, v1_o * 16 + v1_i_init, v2_o * 16 + v2_i_init])
                                                C_reindex[0, v1_o * 16 + v1_i_init, v2_o * 16 + v2_i_init] = T.float16(0)
                                    for ax1_1, ax2_1, ax3_1 in T.grid(16, 16, 16):
                                        with T.block("C"):
                                            v1_i, v2_i, v3_i = T.axis.remap("SSR", [ax1_1, ax2_1, ax3_1])
                                            T.reads(C_reindex[0, v1_o * 16 + v1_i, v2_o * 16 + v2_i], A_reindex_shared_dyn[0, v1_o * 16 + v1_i, v3_o * 16 + v3_i], B_reindex_reindex_shared_dyn[0, v2_o, v3_o, v2_i, v3_i])
                                            T.writes(C_reindex[0, v1_o * 16 + v1_i, v2_o * 16 + v2_i])
                                            C_reindex[0, v1_o * 16 + v1_i, v2_o * 16 + v2_i] = C_reindex[0, v1_o * 16 + v1_i, v2_o * 16 + v2_i] + A_reindex_shared_dyn[0, v1_o * 16 + v1_i, v3_o * 16 + v3_i] * B_reindex_reindex_shared_dyn[0, v2_o, v3_o, v2_i, v3_i]
    for ax0, ax1 in T.grid(16384, 16384):
        with T.block("C_reindex"):
            v0, v1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C_reindex[0, v0, v1])
            T.writes(C[v0, v1])
            C[v0, v1] = C_reindex[0, v0, v1]

mod = tvm.IRModule.from_expr(main)
sch = tvm.tir.Schedule(mod, debug_mask="all")
B_reindex_reindex_reindex = sch.get_block("B_reindex_reindex_reindex")
B_reindex = sch.get_block("B_reindex")
B_decode = sch.get_block("B_decode")
B_reindex_reindex_shared_dyn = sch.get_block("B_reindex_reindex_shared.dyn")

block_local = sch.cache_read(B_reindex_reindex_shared_dyn, 0, "local")
sch.compute_inline(B_reindex_reindex_reindex)
sch.compute_inline(B_reindex)
sch.compute_inline(B_decode)
print(sch.mod)
# dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
# with open("after_memory_rewrite.cu", "+w") as f:
#     f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())