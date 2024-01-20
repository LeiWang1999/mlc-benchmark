import tvm
from tvm.script import ir as I
from tvm.script import tir as T

@T.prim_func
def main(A: T.Buffer((T.int64(128), T.int64(14), T.int64(14), T.int64(256)), "float16"), B: T.Buffer((T.int64(1), T.int64(1), T.int64(256), T.int64(512)), "float16"), conv2d_nhwc: T.Buffer((T.int64(128), T.int64(7), T.int64(7), T.int64(512)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True), "global_symbol": "main"})

    pad_temp = T.alloc_buffer((T.int64(128), T.int64(14), T.int64(14), T.int64(256)), "float16")
    for i0, i1, i2, i3 in T.grid(T.int64(128), T.int64(14), T.int64(14), T.int64(256)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, v_i2, v_i3]
    for nn, yy, xx, ff, ry, rx, rc in T.grid(T.int64(128), T.int64(7), T.int64(7), T.int64(512), T.int64(1), T.int64(1), T.int64(256)):
        with T.block("conv2d_nhwc"):
            v_nn, v_yy, v_xx, v_ff, v_ry, v_rx, v_rc = T.axis.remap("SSSSRRR", [nn, yy, xx, ff, ry, rx, rc])
            T.reads(pad_temp[v_nn, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx, v_rc], B[v_ry, v_rx, v_rc, v_ff])
            T.writes(conv2d_nhwc[v_nn, v_yy, v_xx, v_ff])
            with T.init():
                conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] = T.float16(0)
            conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] = conv2d_nhwc[v_nn, v_yy, v_xx, v_ff] + pad_temp[v_nn, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx, v_rc] * B[v_ry, v_rx, v_rc, v_ff]

mod = tvm.IRModule.from_expr(main)
sch = tvm.tir.Schedule(mod, debug_mask="all")
sch.get_block("conv2d_nhwc")
print(sch.mod)
main_block = sch.get_block("conv2d_nhwc")
sch.reindex(main_block, ("read", 0), skip_simplify=True)
# sch.cache_read(main_block, 1, "global")
print(sch.mod)