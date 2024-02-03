import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.base.utils import apply_and_build, apply_and_build_parallel
import time
from tvm.tir.tensor_intrin.cuda import get_mma_intrin_group

@I.ir_module
class Module:
    @T.prim_func
    def main(a: T.handle, B: T.Buffer((1024, 1024), "float16"), c: T.handle):
        T.func_attr({"tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        m = T.int32()
        A = T.match_buffer(a, (m, 1024), "float16")
        C = T.match_buffer(c, (m, 1024), "float16")
        if m <= 1024:
            Module.main_kernel_opt_m_1024(A.data, B.data, C.data, m)
        else:
            Module.main_kernel_opt_m_1024(A.data, B.data, C.data, m)
    
    @T.prim_func(private=True)
    def main_kernel_opt_m_1024(A_data: T.handle("float16"), B_data: T.handle("float16"), C_data: T.handle("float16"), m: T.int32):
        T.func_attr({"tir.is_global_func": T.bool(True), "tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_reindex_pad_shared = T.alloc_buffer((1, (m + 63) // 64 * 64, 1024), "float16", scope="shared")
        B_reindex_shared = T.alloc_buffer((1, 1024, 1024), "float16", scope="shared")
        A_reindex_pad_shared_warp = T.alloc_buffer((1, (m + 63) // 64 * 4, 64, 32, 8), "float16", scope="warp")
        B_reindex_shared_warp = T.alloc_buffer((1, 64, 64, 32, 8), "float16", scope="warp")
        C_reindex_pad_shared = T.alloc_buffer((1, (m + 63) // 64 * 4, 64, 16, 16), "float16", scope="shared")
        C_reindex_pad_shared_warp = T.alloc_buffer((1, (m + 63) // 64 * 4, 64, 32, 8), "float16", scope="warp")
        A = T.decl_buffer((m, 1024), "float16", data = A_data)
        B = T.decl_buffer((1024, 1024), "float16", data = B_data)
        C = T.decl_buffer((m, 1024), "float16", data = C_data)
        
        for ax0 in T.thread_binding(1, thread="blockIdx.z"):
            for ax1_0_0_ax2_0_0_fused in T.thread_binding((m + 63) // 64, thread="blockIdx.y"):
                for ax1_0_1_ax2_0_1_fused in T.thread_binding(16, thread="blockIdx.x"):
                    for ax1_0_2 in T.thread_binding(2, thread="threadIdx.y"):
                        for ax2_0_2 in T.thread_binding(2, thread="threadIdx.z"):
                            for ax1_0_3_init, ax2_0_3_init in T.grid(2, 2):
                                with T.block("B_o_init"):
                                    v0_o = T.axis.spatial(1, ax0)
                                    v1_o = T.axis.spatial((m + 63) // 64 * 4, ax1_0_0_ax2_0_0_fused * 4 + ax1_0_2 * 2 + ax1_0_3_init)
                                    v2_o = T.axis.spatial(64, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax2_0_3_init)
                                    T.reads()
                                    T.writes(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                    with T.block("B_init_o"):
                                        v1_i_init_o = T.axis.spatial(1, 0)
                                        v2_i_init_o = T.axis.spatial(1, 0)
                                        T.reads()
                                        T.writes(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                        C_warp = T.match_buffer(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                        for tx in T.thread_binding(32, thread="threadIdx.x"):
                                            T.mma_fill("float16", 8, C_warp.data, C_warp.elem_offset)
                            for ax3_0_0 in T.serial(32, annotations={"software_pipeline_async_stages": [0], "software_pipeline_order": [0, 1, 2], "software_pipeline_stage": [0, 0, 1]}):
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("A_reindex_pad_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial((m + 63) // 64 * 64, ax1_0_0_ax2_0_0_fused * 64 + (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 32)
                                                        v2 = T.axis.spatial(1024, ax3_0_0 * 32 + (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 32)
                                                        T.reads(A[v1, v2])
                                                        T.writes(A_reindex_pad_shared[v0, v1, v2])
                                                        T.block_attr({"permuted_layout": 1})
                                                        A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, A[v1, v2], T.float16(0))
                                for ax0_ax1_ax2_fused_0 in T.thread_binding(2, thread="threadIdx.y"):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(2, thread="threadIdx.z"):
                                        for ax0_ax1_ax2_fused_2 in T.unroll(2, annotations={"pragma_unroll_explicit": 0}):
                                            for ax0_ax1_ax2_fused_3 in T.thread_binding(32, thread="threadIdx.x"):
                                                for ax0_ax1_ax2_fused_4 in T.vectorized(8):
                                                    with T.block("B_reindex_shared"):
                                                        v0 = T.axis.spatial(1, 0)
                                                        v1 = T.axis.spatial(1024, ax1_0_1_ax2_0_1_fused * 64 + (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) // 32)
                                                        v2 = T.axis.spatial(1024, ax3_0_0 * 32 + (ax0_ax1_ax2_fused_0 * 1024 + ax0_ax1_ax2_fused_1 * 512 + ax0_ax1_ax2_fused_2 * 256 + ax0_ax1_ax2_fused_3 * 8 + ax0_ax1_ax2_fused_4) % 32)
                                                        T.reads(B[v1, v2])
                                                        T.writes(B_reindex_shared[v0, v1, v2])
                                                        T.block_attr({"permuted_layout": 1})
                                                        B_reindex_shared[v0, v1, v2] = B[v1, v2]
                                for ax3_0_1 in range(2):
                                    for ax0_0, ax1_0 in T.grid(2, 1):
                                        with T.block("A_reindex_pad_shared_warp_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(4 * ((m + 63) // 64), ax1_0_0_ax2_0_0_fused * 4 + ax1_0_2 * 2 + ax0_0)
                                            v2_o = T.axis.spatial(64, ax3_0_0 * 2 + ax3_0_1 + ax1_0)
                                            T.reads(A_reindex_pad_shared[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(A_reindex_pad_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 1})
                                            warp = T.match_buffer(A_reindex_pad_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(A_reindex_pad_shared[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * (tx % 16) + 8 * (tx // 16))
                                    for ax0_0, ax1_0 in T.grid(2, 1):
                                        with T.block("B_reindex_shared_warp_o"):
                                            v0_o = T.axis.spatial(1, 0)
                                            v1_o = T.axis.spatial(64, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax0_0)
                                            v2_o = T.axis.spatial(64, ax3_0_0 * 2 + ax3_0_1 + ax1_0)
                                            T.reads(B_reindex_shared[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16])
                                            T.writes(B_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                            T.block_attr({"permuted_layout": 1})
                                            warp = T.match_buffer(B_reindex_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                            shared = T.match_buffer(B_reindex_shared[v0_o, v1_o * 16:v1_o * 16 + 16, v2_o * 16:v2_o * 16 + 16], (16, 16), "float16", strides=("shared_s0", "shared_s1"), scope="shared", offset_factor=16)
                                            for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", warp.data, warp.elem_offset + 8 * tx, T.tvm_access_ptr(T.type_annotation("float16"), shared.data, shared.elem_offset, shared.strides[0] * 16, 1), shared.strides[0] * 8 * (tx // 16) + shared.strides[0] * (tx % 8) + 8 * (tx % 16 // 8))
                                    for ax1_0_3, ax2_0_3 in T.grid(2, 2):
                                        with T.block("B_o_update"):
                                            v0_o = T.axis.spatial(1, ax0)
                                            v1_o = T.axis.spatial((m + 63) // 64 * 4, ax1_0_0_ax2_0_0_fused * 4 + ax1_0_2 * 2 + ax1_0_3)
                                            v2_o = T.axis.spatial(64, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax2_0_3)
                                            v3_o = T.axis.reduce(64, ax3_0_0 * 2 + ax3_0_1)
                                            T.reads(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_pad_shared_warp[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:8])
                                            T.writes(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                            with T.block("B_o"):
                                                v1_i_o = T.axis.spatial(1, 0)
                                                v2_i_o = T.axis.spatial(1, 0)
                                                v3_i_o = T.axis.reduce(1, 0)
                                                T.reads(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8], A_reindex_pad_shared_warp[0, v1_o, v3_o, 0:32, 0:8], B_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:8])
                                                T.writes(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8])
                                                A_1 = T.match_buffer(A_reindex_pad_shared_warp[0, v1_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                B_1 = T.match_buffer(B_reindex_shared_warp[0, v2_o, v3_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                C_1 = T.match_buffer(C_reindex_pad_shared_warp[0, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=16)
                                                for tx in T.thread_binding(32, thread="threadIdx.x"):
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8, C_1.data, C_1.elem_offset + tx * 8, T.bool(False))
                                                    T.ptx_mma("float16", "m16n8k16", "row", "col", "fp16", "fp16", "fp16", A_1.data, A_1.elem_offset + tx * 8, B_1.data, B_1.elem_offset + tx * 8 + 4, C_1.data, C_1.elem_offset + tx * 8 + 4, T.bool(False))
                            for ax0_1, ax1, ax2_0, ax3_0 in T.grid(2, 2, 1, 1):
                                with T.block("C_reindex_pad_shared_warp_o"):
                                    v0_o = T.axis.spatial(1, 0)
                                    v1_o = T.axis.spatial(4 * ((m + 63) // 64), ax1_0_0_ax2_0_0_fused * 4 + ax1_0_2 * 2 + ax0_1)
                                    v2_o = T.axis.spatial(64, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax1)
                                    v3_o, v4_o = T.axis.remap("SS", [ax2_0, ax3_0])
                                    T.reads(C_reindex_pad_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8])
                                    T.writes(C_reindex_pad_shared[v0_o, v1_o, v2_o, 0:16, 0:16])
                                    C_warp = T.match_buffer(C_reindex_pad_shared_warp[v0_o, v1_o, v2_o, 0:32, 0:8], (32, 8), "float16", scope="warp", offset_factor=1)
                                    C_1 = T.match_buffer(C_reindex_pad_shared[v0_o, v1_o, v2_o, 0:16, 0:16], (16, 16), "float16", strides=("C_s0", "C_s1"), scope="shared", offset_factor=1)
                                    for tx in T.thread_binding(32, thread="threadIdx.x"):
                                        T.mma_store("float16", 16, 16, T.tvm_access_ptr(T.type_annotation("float16"), C_1.data, C_1.elem_offset, C_1.strides[0] * 16, 2), C_warp.data, C_warp.elem_offset, C_1.strides[0])
                                for ax0_ax1_ax2_ax3_ax4_fused_0 in T.unroll(1, annotations={"pragma_unroll_explicit": 0}):
                                    for ax0_ax1_ax2_ax3_ax4_fused_1 in T.thread_binding(32, thread="threadIdx.x"):
                                        for ax0_ax1_ax2_ax3_ax4_fused_2 in T.vectorized(8):
                                            with T.block("C_reindex_pad_shared"):
                                                v0 = T.axis.spatial(1, 0)
                                                v1 = T.axis.spatial((m + 63) // 64 * 4, ax1_0_0_ax2_0_0_fused * 4 + ax1_0_2 * 2 + ax0_1)
                                                v2 = T.axis.spatial(64, ax1_0_1_ax2_0_1_fused * 4 + ax2_0_2 * 2 + ax1)
                                                v3 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_ax4_fused_0 * 256 + ax0_ax1_ax2_ax3_ax4_fused_1 * 8 + ax0_ax1_ax2_ax3_ax4_fused_2) // 16)
                                                v4 = T.axis.spatial(16, (ax0_ax1_ax2_ax3_ax4_fused_0 * 256 + ax0_ax1_ax2_ax3_ax4_fused_1 * 8 + ax0_ax1_ax2_ax3_ax4_fused_2) % 16)
                                                T.reads(C_reindex_pad_shared[v0, v1, v2, v3, v4])
                                                T.writes(C[v3 + v1 * 16, v4 + v2 * 16])
                                                if v1 * 16 + v3 < m:
                                                    C[v3 + v1 * 16, v4 + v2 * 16] = C_reindex_pad_shared[v0, v1, v2, v3, v4]
 
sch = tvm.tir.Schedule(Module, debug_mask="all")
dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
A = tvm.nd.array(np.random.rand(1024, 1024).astype("float16"), device=tvm.cuda())
B = tvm.nd.array(np.random.rand(1024, 1024).astype("float16"), device=tvm.cuda())
C = tvm.nd.array(np.zeros((1024, 1024), dtype="float16"), device=tvm.cuda())
dense_relu_0_rt_mod(A, B, C)
print(C)
with open("after_memory_rewrite.cu", "+w") as f:
    f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())
