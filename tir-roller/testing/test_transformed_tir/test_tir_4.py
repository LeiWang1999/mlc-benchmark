import tvm
from tvm.script import ir as I
from tvm.script import tir as T

# from tvm.script import ir as I
# from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((T.int64(128), T.int64(224), T.int64(224), T.int64(3)), "float16"), B: T.Buffer((T.int64(64), T.int64(7), T.int64(7), T.int64(3)), "float16"), conv2d_nhwc: T.Buffer((T.int64(128), T.int64(112), T.int64(112), T.int64(64)), "float16")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp_reindex_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1605632), T.int64(160)), "float16", scope="shared.dyn")
    B_reindex_reindex_pad_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(160)), "float16", scope="shared.dyn")
    pad_temp_reindex_reindex_pad_shared_dyn_wmma_matrix_a = T.alloc_buffer((T.int64(1), T.int64(1605632), T.int64(160)), "float16", scope="wmma.matrix_a")
    B_reindex_reindex_pad_shared_dyn_wmma_matrix_b = T.alloc_buffer((T.int64(1), T.int64(64), T.int64(160)), "float16", scope="wmma.matrix_b")
    conv2d_nhwc_reindex_reindex_shared_dyn = T.alloc_buffer((T.int64(1), T.int64(1605632), T.int64(64)), "float16", scope="shared.dyn")
    conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator = T.alloc_buffer((T.int64(1), T.int64(1605632), T.int64(64)), "float16", scope="wmma.accumulator")
    for ax0 in T.thread_binding(T.int64(1), thread="blockIdx.z"):
        for ax1_0_0_ax2_0_0_fused in T.thread_binding(T.int64(100352), thread="blockIdx.x"):
            for ax1_0_1_ax2_0_1_fused in T.thread_binding(T.int64(1), thread="blockIdx.y"):
                for ax2_0_2_ax1_0_2_fused in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                    for ax1_0_3_init, ax2_0_3_init in T.grid(T.int64(1), T.int64(1)):
                        with T.block("conv2d_nhwc_o_init"):
                            v0_o = T.axis.spatial(T.int64(1), ax0)
                            v1_o = T.axis.spatial(T.int64(100352), ax1_0_0_ax2_0_0_fused + ax1_0_3_init)
                            v2_o = T.axis.spatial(T.int64(4), ax2_0_2_ax1_0_2_fused + ax2_0_3_init)
                            T.reads()
                            T.writes(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                            with T.block("conv2d_nhwc_init_o"):
                                v1_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                v2_i_init_o = T.axis.spatial(T.int64(1), T.int64(0))
                                T.reads()
                                T.writes(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                C = T.match_buffer(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                T.tvm_fill_fragment(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.float32(0))
                    for ax3_0_0 in T.serial(T.int64(5), annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                        for ax0_ax1_fused_0 in range(T.int64(1)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                        with T.block("pad_temp_reindex_reindex_pad_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(1605632), ax1_0_0_ax2_0_0_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(512) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) // T.int64(32))
                                            v2 = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(512) + ax0_ax1_fused_1 * T.int64(128) + ax0_ax1_fused_2 * T.int64(4) + ax0_ax1_fused_3) % T.int64(32))
                                            T.reads(A[v1 // T.int64(12544), v1 // T.int64(112) % T.int64(112) * T.int64(2) + v2 // T.int64(21) - T.int64(3), v1 % T.int64(112) * T.int64(2) + v2 // T.int64(3) % T.int64(7) - T.int64(3), v2 % T.int64(3)])
                                            T.writes(pad_temp_reindex_reindex_pad_shared_dyn[v0, v1, v2])
                                            T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]]})
                                            pad_temp_reindex_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v2 < T.int64(147), T.if_then_else(T.int64(3) <= v1 // T.int64(112) % T.int64(112) * T.int64(2) + v2 // T.int64(21) and v1 // T.int64(112) % T.int64(112) * T.int64(2) + v2 // T.int64(21) < T.int64(227) and T.int64(3) <= v1 % T.int64(112) * T.int64(2) + v2 // T.int64(3) % T.int64(7) and v1 % T.int64(112) * T.int64(2) + v2 // T.int64(3) % T.int64(7) < T.int64(227), A[v1 // T.int64(12544), v1 // T.int64(112) % T.int64(112) * T.int64(2) + v2 // T.int64(21) - T.int64(3), v1 % T.int64(112) * T.int64(2) + v2 // T.int64(3) % T.int64(7) - T.int64(3), v2 % T.int64(3)], T.float16(0)), T.float16(0))
                        for ax0_ax1_fused_0 in range(T.int64(4)):
                            for ax0_ax1_fused_1 in T.thread_binding(T.int64(4), thread="threadIdx.y"):
                                for ax0_ax1_fused_2 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                                    for ax0_ax1_fused_3 in T.vectorized(T.int64(4)):
                                        with T.block("B_reindex_reindex_pad_shared.dyn"):
                                            v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                            v1 = T.axis.spatial(T.int64(64), (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) // T.int64(32))
                                            v2 = T.axis.spatial(T.int64(160), ax3_0_0 * T.int64(32) + (ax0_ax1_fused_0 * T.int64(1024) + ax0_ax1_fused_1 * T.int64(256) + ax0_ax1_fused_2 * T.int64(8) + ax0_ax1_fused_3) % T.int64(32))
                                            T.reads(B[v1, v2 // T.int64(21), v2 // T.int64(3) % T.int64(7), v2 % T.int64(3)])
                                            T.writes(B_reindex_reindex_pad_shared_dyn[v0, v1, v2])
                                            T.block_attr({"buffer_dim_align": [[0, 1, 16, 8]]})
                                            B_reindex_reindex_pad_shared_dyn[v0, v1, v2] = T.if_then_else(v2 < T.int64(147), B[v1, v2 // T.int64(21), v2 // T.int64(3) % T.int64(7), v2 % T.int64(3)], T.float16(0))
                        for ax3_0_1 in range(T.int64(2)):
                            for ax0_0 in T.unroll(T.int64(1)):
                                for ax1_0 in T.unroll(T.int64(1)):
                                    with T.block("pad_temp_reindex_reindex_pad_shared.dyn_wmma.matrix_a_o"):
                                        v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1_o = T.axis.spatial(T.int64(100352), ax1_0_0_ax2_0_0_fused + ax0_0)
                                        v2_o = T.axis.spatial(T.int64(10), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                        T.reads(pad_temp_reindex_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(pad_temp_reindex_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        A_1 = T.match_buffer(pad_temp_reindex_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                        C = T.match_buffer(pad_temp_reindex_reindex_pad_shared_dyn_wmma_matrix_a[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_a", offset_factor=16)
                                        T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "row_major")
                            for ax0_0 in T.unroll(T.int64(1)):
                                for ax1_0 in T.unroll(T.int64(1)):
                                    with T.block("B_reindex_reindex_pad_shared.dyn_wmma.matrix_b_o"):
                                        v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v1_o = T.axis.spatial(T.int64(4), ax2_0_2_ax1_0_2_fused + ax0_0)
                                        v2_o = T.axis.spatial(T.int64(10), ax3_0_0 * T.int64(2) + ax3_0_1 + ax1_0)
                                        T.reads(B_reindex_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        T.writes(B_reindex_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        A_1 = T.match_buffer(B_reindex_reindex_pad_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="shared.dyn", offset_factor=16)
                                        C = T.match_buffer(B_reindex_reindex_pad_shared_dyn_wmma_matrix_b[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.matrix_b", offset_factor=16)
                                        T.tvm_load_matrix_sync(C.data, 16, 16, 16, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), A_1.data, A_1.elem_offset, A_1.strides[0] * T.int64(16), 1), A_1.strides[0], "col_major")
                            for ax1_0_3, ax2_0_3 in T.grid(T.int64(1), T.int64(1)):
                                with T.block("conv2d_nhwc_o_update"):
                                    v0_o = T.axis.spatial(T.int64(1), ax0)
                                    v1_o = T.axis.spatial(T.int64(100352), ax1_0_0_ax2_0_0_fused + ax1_0_3)
                                    v2_o = T.axis.spatial(T.int64(4), ax2_0_2_ax1_0_2_fused + ax2_0_3)
                                    v3_o = T.axis.reduce(T.int64(10), ax3_0_0 * T.int64(2) + ax3_0_1)
                                    T.reads(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], pad_temp_reindex_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], B_reindex_reindex_pad_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                    T.writes(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                    with T.block("conv2d_nhwc_o"):
                                        v1_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v2_i_o = T.axis.spatial(T.int64(1), T.int64(0))
                                        v3_i_o = T.axis.reduce(T.int64(1), T.int64(0))
                                        T.reads(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], pad_temp_reindex_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], B_reindex_reindex_pad_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)])
                                        T.writes(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                                        A_1 = T.match_buffer(pad_temp_reindex_reindex_pad_shared_dyn_wmma_matrix_a[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.matrix_a", offset_factor=16)
                                        B_1 = T.match_buffer(B_reindex_reindex_pad_shared_dyn_wmma_matrix_b[T.int64(0), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16), v3_o * T.int64(16):v3_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("B_s0", "B_s1"), scope="wmma.matrix_b", offset_factor=16)
                                        C = T.match_buffer(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[T.int64(0), v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="wmma.accumulator", offset_factor=16)
                                        T.tvm_mma_sync(C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16), A_1.data, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), B_1.data, B_1.elem_offset // B_1.strides[0] // T.int64(16) * (B_1.strides[0] // T.int64(16)) + B_1.elem_offset % B_1.strides[0] // T.int64(16), C.data, C.elem_offset // C.strides[0] // T.int64(16) * (C.strides[0] // T.int64(16)) + C.elem_offset % C.strides[0] // T.int64(16))
                    for ax0_0, ax1_0 in T.grid(T.int64(1), T.int64(1)):
                        with T.block("conv2d_nhwc_reindex_reindex_shared.dyn_wmma.accumulator_o"):
                            v0_o = T.axis.spatial(T.int64(1), T.int64(0))
                            v1_o = T.axis.spatial(T.int64(100352), ax1_0_0_ax2_0_0_fused + ax0_0)
                            v2_o = T.axis.spatial(T.int64(4), ax2_0_2_ax1_0_2_fused + ax1_0)
                            T.reads(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                            T.writes(conv2d_nhwc_reindex_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)])
                            A_1 = T.match_buffer(conv2d_nhwc_reindex_reindex_shared_dyn_wmma_accumulator[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("A_s0", "A_s1"), scope="wmma.accumulator", offset_factor=16)
                            C = T.match_buffer(conv2d_nhwc_reindex_reindex_shared_dyn[v0_o, v1_o * T.int64(16):v1_o * T.int64(16) + T.int64(16), v2_o * T.int64(16):v2_o * T.int64(16) + T.int64(16)], (T.int64(16), T.int64(16)), "float16", strides=("C_s0", "C_s1"), scope="shared.dyn", offset_factor=16)
                            T.tvm_store_matrix_sync(A_1.data, 16, 16, 16, A_1.elem_offset // A_1.strides[0] // T.int64(16) * (A_1.strides[0] // T.int64(16)) + A_1.elem_offset % A_1.strides[0] // T.int64(16), T.tvm_access_ptr(T.type_annotation("float16"), C.data, C.elem_offset, C.strides[0] * T.int64(16), 2), C.strides[0], "row_major")
                    for ax0_ax1_fused_0 in range(T.int64(1)):
                        for ax0_ax1_fused_1 in T.thread_binding(T.int64(32), thread="threadIdx.x"):
                            for ax0_ax1_fused_2 in T.vectorized(T.int64(8)):
                                with T.block("conv2d_nhwc_reindex_reindex_shared.dyn"):
                                    v0 = T.axis.spatial(T.int64(1), T.int64(0))
                                    v1 = T.axis.spatial(T.int64(1605632), ax1_0_0_ax2_0_0_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(8) + ax0_ax1_fused_2) // T.int64(16))
                                    v2 = T.axis.spatial(T.int64(64), ax2_0_2_ax1_0_2_fused * T.int64(16) + (ax0_ax1_fused_0 * T.int64(256) + ax0_ax1_fused_1 * T.int64(8) + ax0_ax1_fused_2) % T.int64(16))
                                    T.reads(conv2d_nhwc_reindex_reindex_shared_dyn[v0, v1, v2])
                                    T.writes(conv2d_nhwc[v1 // T.int64(12544), v1 // T.int64(112) % T.int64(112), v1 % T.int64(112), v2])
                                    T.block_attr({"buffer_dim_align": [[0, 1, 16, 4]]})
                                    conv2d_nhwc[v1 // T.int64(12544), v1 // T.int64(112) % T.int64(112), v1 % T.int64(112), v2] = conv2d_nhwc_reindex_reindex_shared_dyn[v0, v1, v2]
mod = tvm.IRModule.from_expr(main)
sch = tvm.tir.Schedule(mod, debug_mask="all")
rt_mod = tvm.build(sch.mod, target="cuda")

with open("after_memory_rewrite.cu", "+w") as f:
    f.write(rt_mod.imported_modules[0].get_source())

n, h, w, ic, oc, kh, kw, stride, padding = 128, 224, 224, 3, 64, 7, 7, 2, 3

import numpy as np
device = tvm.device("cuda", 0)
dtype = "float16"
A = np.random.uniform(size=(n, h, w, ic)).astype(dtype)
W = np.random.uniform(size=(oc, kh, kw, ic)).astype(dtype)

tvm_a = tvm.nd.array(A, device)
tvm_b = tvm.nd.array(W, device)
tvm_c = tvm.nd.array(np.zeros((n, h // stride, w // stride, oc), dtype=dtype), device)
rt_mod(tvm_a, tvm_b, tvm_c)
print(tvm_c)
