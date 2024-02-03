import numpy as np
import tvm
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags
from tvm.dlight.base.utils import apply_and_build, apply_and_build_parallel
import time

@T.prim_func
def original(a: T.handle, b: T.handle, c: T.handle):
    m = T.int32()
    T.func_attr({"global_symbol": "main", "tir.noalias": True})
    A = T.match_buffer(a, [m, 1024], dtype="float32")
    B = T.match_buffer(b, [1024, 1024], dtype="float32")
    C = T.match_buffer(c, [m, 1024], dtype="float32")

    for i, j, k in T.grid(m, 1024, 1024):
        with T.block("B"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = tvm.tir.const(0, "float32")
            C[vi, vj] = C[vi, vj] + A[vi, vk].astype("float32") * B[vj, vk].astype(
                "float32"
            )

@T.prim_func
def matmul_opt_m_1024(a: T.handle, B: T.Buffer((1024, 1024), "float32"), c: T.handle):
    T.func_attr({"opt_shapes": {"m": 1024}, "tir.noalias": T.bool(True)})
    m = T.int32()
    A = T.match_buffer(a, (m, 1024))
    C = T.match_buffer(c, (m, 1024))
    # with T.block("root"):
    C_reindex_pad_local = T.alloc_buffer((1, (m + 31) // 32 * 32, 1024), scope="local")
    A_reindex_pad_shared = T.alloc_buffer((1, (m + 31) // 32 * 32, 1024), scope="shared")
    A_reindex_pad_shared_local = T.alloc_buffer((1, (m + 31) // 32 * 32, 1024), scope="local")
    B_reindex_shared = T.alloc_buffer((1, 1024, 1024), scope="shared")
    B_reindex_shared_local = T.alloc_buffer((1, 1024, 1024), scope="local")
    for ax0_ax1_0_fused in T.thread_binding((m + 31) // 32, thread="blockIdx.y"):
        for ax2_0 in T.thread_binding(32, thread="blockIdx.x"):
            for ax1_1 in T.thread_binding(2, thread="vthread.y"):
                for ax2_1 in T.thread_binding(2, thread="vthread.x"):
                    for ax1_2 in T.thread_binding(8, thread="threadIdx.y"):
                        for ax2_2 in T.thread_binding(8, thread="threadIdx.x"):
                            for ax1_3_init, ax2_3_init in T.grid(2, 2):
                                with T.block("B_init"):
                                    v0 = T.axis.spatial(1, 0)
                                    v1 = T.axis.spatial((m + 31) // 32 * 32, ax0_ax1_0_fused * 32 + ax1_1 * 16 + ax1_2 * 2 + ax1_3_init)
                                    v2 = T.axis.spatial(1024, ax2_0 * 32 + ax2_1 * 16 + ax2_2 * 2 + ax2_3_init)
                                    T.reads()
                                    T.writes(C_reindex_pad_local[0, v1, v2])
                                    C_reindex_pad_local[0, v1, v2] = T.float32(0)
                            for ax3_0 in range(32):
                                for ax0_ax1_ax2_fused_0 in range(4):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_2 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_3 in T.vectorized(4):
                                                with T.block("A_reindex_pad_shared"):
                                                    v0 = T.axis.spatial(1, 0)
                                                    v1 = T.axis.spatial((m + 31) // 32 * 32, ax0_ax1_0_fused * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 32 + ax0_ax1_ax2_fused_2 * 4 + ax0_ax1_ax2_fused_3) // 32)
                                                    v2 = T.axis.spatial(1024, ax3_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 32 + ax0_ax1_ax2_fused_2 * 4 + ax0_ax1_ax2_fused_3) % 32)
                                                    T.reads(A[v1, v2])
                                                    T.writes(A_reindex_pad_shared[v0, v1, v2])
                                                    A_reindex_pad_shared[v0, v1, v2] = T.if_then_else(v1 < m, A[v1, v2], T.float32(0))
                                for ax0_ax1_ax2_fused_0 in range(4):
                                    for ax0_ax1_ax2_fused_1 in T.thread_binding(8, thread="threadIdx.y"):
                                        for ax0_ax1_ax2_fused_2 in T.thread_binding(8, thread="threadIdx.x"):
                                            for ax0_ax1_ax2_fused_3 in range(4):
                                                with T.block("B_reindex_shared"):
                                                    v0 = T.axis.spatial(1, 0)
                                                    v1 = T.axis.spatial(1024, ax3_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 32 + ax0_ax1_ax2_fused_2 * 4 + ax0_ax1_ax2_fused_3) // 32)
                                                    v2 = T.axis.spatial(1024, ax2_0 * 32 + (ax0_ax1_ax2_fused_0 * 256 + ax0_ax1_ax2_fused_1 * 32 + ax0_ax1_ax2_fused_2 * 4 + ax0_ax1_ax2_fused_3) % 32)
                                                    T.reads(B[v2, v1])
                                                    T.writes(B_reindex_shared[v0, v1, v2])
                                                    B_reindex_shared[v0, v1, v2] = B[v2, v1]
                                for ax3_1 in range(32):
                                    for ax0, ax1_ax2_fused_0 in T.grid(1, 1):
                                        for ax1_ax2_fused_1 in T.vectorized(4):
                                            with T.block("A_reindex_pad_shared_local"):
                                                v0 = T.axis.spatial(1, ax0)
                                                v1 = T.axis.spatial((m + 31) // 32 * 32, ax0_ax1_0_fused * 32 + ax1_1 * 16 + ax1_2 * 2 + (ax1_ax2_fused_0 * 4 + ax1_ax2_fused_1))
                                                v2 = T.axis.spatial(1024, ax3_0 * 32 + ax3_1)
                                                T.where(ax1_ax2_fused_0 * 4 + ax1_ax2_fused_1 < 2)
                                                T.reads(A_reindex_pad_shared[v0, v1, v2])
                                                T.writes(A_reindex_pad_shared_local[v0, v1, v2])
                                                A_reindex_pad_shared_local[v0, v1, v2] = A_reindex_pad_shared[v0, v1, v2]
                                    for ax0, ax1_ax2_fused_0 in T.grid(1, 1):
                                        for ax1_ax2_fused_1 in T.vectorized(4):
                                            with T.block("B_reindex_shared_local"):
                                                v0 = T.axis.spatial(1, ax0)
                                                v1 = T.axis.spatial(1024, ax3_0 * 32 + ax3_1)
                                                v2 = T.axis.spatial(1024, ax2_0 * 32 + ax2_1 * 16 + ax2_2 * 2 + (ax1_ax2_fused_0 * 4 + ax1_ax2_fused_1))
                                                T.where(ax1_ax2_fused_0 * 4 + ax1_ax2_fused_1 < 2)
                                                T.reads(B_reindex_shared[v0, v1, v2])
                                                T.writes(B_reindex_shared_local[v0, v1, v2])
                                                B_reindex_shared_local[v0, v1, v2] = B_reindex_shared[v0, v1, v2]
                                    for ax1_3, ax2_3 in T.grid(2, 2):
                                        with T.block("B_update"):
                                            v0 = T.axis.spatial(1, 0)
                                            v1 = T.axis.spatial((m + 31) // 32 * 32, ax0_ax1_0_fused * 32 + ax1_1 * 16 + ax1_2 * 2 + ax1_3)
                                            v2 = T.axis.spatial(1024, ax2_0 * 32 + ax2_1 * 16 + ax2_2 * 2 + ax2_3)
                                            v3 = T.axis.reduce(1024, ax3_0 * 32 + ax3_1)
                                            T.reads(C_reindex_pad_local[0, v1, v2], A_reindex_pad_shared_local[0, v1, v3], B_reindex_shared_local[0, v3, v2])
                                            T.writes(C_reindex_pad_local[0, v1, v2])
                                            C_reindex_pad_local[0, v1, v2] = C_reindex_pad_local[0, v1, v2] + A_reindex_pad_shared_local[0, v1, v3] * B_reindex_shared_local[0, v3, v2]
                            for ax0, ax1_ax2_fused_0 in T.grid(1, 1):
                                for ax1_ax2_fused_1 in T.vectorized(4):
                                    with T.block("C_reindex_pad_local"):
                                        v0 = T.axis.spatial(1, ax0)
                                        v1 = T.axis.spatial((m + 31) // 32 * 32, ax0_ax1_0_fused * 32 + ax1_1 * 16 + ax1_2 * 2 + (ax1_ax2_fused_0 * 4 + ax1_ax2_fused_1) // 2)
                                        v2 = T.axis.spatial(1024, ax2_0 * 32 + ax2_1 * 16 + ax2_2 * 2 + (ax1_ax2_fused_0 * 4 + ax1_ax2_fused_1) % 2)
                                        T.reads(C_reindex_pad_local[v0, v1, v2])
                                        T.writes(C[v1, v2])
                                        if v1 < m:
                                            C[v1, v2] = C_reindex_pad_local[v0, v1, v2]