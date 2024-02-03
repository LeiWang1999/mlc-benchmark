import tvm
from tvm.script import ir as I
from tvm.script import tir as T
from tvm.tir.tensor_intrin.cuda import get_mma_intrin_group

@T.prim_func
def fused_decode5_fused_matmul6_silu1(
    lv1611: T.Buffer((T.int64(512), T.int64(11008)), "uint32"),
    lv1612: T.Buffer((T.int64(128), T.int64(11008)), "float16"),
    lv1622: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    ),
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_decode_intermediate = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    )
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1611[v_i // T.int64(8), v_j], lv1612[v_i // T.int64(32), v_j])
            T.writes(var_decode_intermediate[v_i, v_j])
            var_decode_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1611[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1612[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(11008), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1622[v_i0, v_i1, v_k], var_decode_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1622[v_i0, v_i1, v_k] * var_decode_intermediate[v_k, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(
                var_matmul_intermediate[v_i0, v_i1, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2],
                compute[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
                * compute[v_ax0, v_ax1, v_ax2]
            )

def sch_fused_decode5_fused_matmul6_silu1(func):
    sch = tvm.tir.Schedule(func)
    b0 = sch.get_block(name="decode", func_name="main")
    b1 = sch.get_block(name="matmul", func_name="main")
    l2, l3, l4, l5 = sch.get_loops(block=b1)
    l6 = sch.fuse(l2, l3, l4, preserve_unit_iters=True)
    v7, v8, v9 = sch.sample_perfect_tile(
        loop=l6, n=3, max_innermost_factor=4, decision=[43, 64, 4]
    )
    l10, l11, l12 = sch.split(loop=l6, factors=[v7, v8, v9], preserve_unit_iters=True)
    v13, v14, v15 = sch.sample_perfect_tile(
        loop=l5, n=3, max_innermost_factor=8, decision=[128, 4, 8]
    )
    l16, l17, l18 = sch.split(
        loop=l5, factors=[v13, v14, v15], preserve_unit_iters=True
    )
    sch.reorder(l10, l11, l16, l17, l18, l12)
    sch.bind(loop=l10, thread_axis="blockIdx.x")
    sch.bind(loop=l11, thread_axis="threadIdx.x")
    sch.compute_inline(block=b0)
    b19 = sch.cache_write(block=b1, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=b19, loop=l11, preserve_unit_loops=True, index=-1)
    b20 = sch.cache_read(block=b1, read_buffer_index=1, storage_scope="local")
    b21 = sch.cache_read(block=b1, read_buffer_index=2, storage_scope="local")
    b22 = sch.cache_read(block=b1, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b22, loop=l11, preserve_unit_loops=True, index=-1)
    v23 = sch.sample_categorical(
        candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3
    )
    sch.annotate(
        block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch", ann_val=v23
    )
    sch.compute_at(block=b20, loop=l17, preserve_unit_loops=True, index=-1)
    sch.compute_at(block=b21, loop=l16, preserve_unit_loops=True, index=-1)
    l24, l25, l26, l27, l28, l29 = sch.get_loops(block=b20)
    sch.vectorize(loop=l29)
    l30, l31, l32, l33, l34 = sch.get_loops(block=b21)
    sch.vectorize(loop=l34)
    l35, l36, l37, l38, l39 = sch.get_loops(block=b19)
    sch.vectorize(loop=l39)
    sch.vectorize(loop=l12)
    b40 = sch.decompose_reduction(block=b1, loop=l16)
    b41 = sch.get_block(name="compute", func_name="main")
    sch.compute_inline(block=b41)
    # print(sch.mod)
    b42 = sch.get_block(name="T_multiply", func_name="main")
    sch.reverse_compute_inline(block=b42)
    print(sch.mod)
    sch.enter_postproc()
    sch.unannotate(block_or_loop=b22, ann_key="meta_schedule.cooperative_fetch")
    l43, l44, l45, l46, l47 = sch.get_loops(block=b22)
    l48, l49, l50 = sch.split(loop=l47, factors=[None, 64, 8], preserve_unit_iters=True)
    sch.vectorize(loop=l50)
    sch.bind(loop=l49, thread_axis="threadIdx.x")
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)

sch_fused_decode5_fused_matmul6_silu1(fused_decode5_fused_matmul6_silu1)
# sch = tvm.tir.Schedule(mod, debug_mask="all")
# dense_relu_0_rt_mod = tvm.build(sch.mod, target="cuda")
# with open("after_memory_rewrite.cu", "+w") as f:
#     f.write(dense_relu_0_rt_mod.imported_modules[0].get_source())
    