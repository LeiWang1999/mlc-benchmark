import numpy as np
import tvm
import time
from tvm.script import tir as T
from tvm.dlight.base.roller.policy import DefaultPolicy
from tvm.dlight.base.roller.policy.default import PrimFuncNode
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import GeneralReduction, Matmul
from tvm.dlight.gpu import Fallback
from tvm.dlight.base.utils import apply_and_build_parallel, apply_and_build
from tvm import te, tir
from tvm.dlight.base.analysis import normalize_with_tensorcore


def conv2d_nhwc_hwnc(n, f, h, w, c, kh, kw, s, d, p, in_dtype="float16", out_dtype="float16"):
    A = te.placeholder((n, h, w, c), name="input", dtype=in_dtype)
    B = te.placeholder((kh, kw, c, f), name="weight", dtype=in_dtype)

    pad_shape = (n, h + 2 * p, w + 2 * p, c)
    pad_value = tir.const(0.0, A.dtype)
    pad = te.compute(
        pad_shape,
        lambda n, h, w, c: te.if_then_else(
            tir.all(
                h >= p,
                w >= p,
                h < pad_shape[1] - p,
                w < pad_shape[2] - p,
            ),
            A[n, h - p, w - p, c],
            pad_value,
        ),
        name="pad",
    )
    kernel_h, kernel_w = kh, kw
    stride_h, stride_w = s, s
    dilation_h, dilation_w = d, d
    out_h = (h + 2 * p - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    out_w = (w + 2 * p - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1
    out_shape = (n, out_h, out_w, f)
    kh = te.reduce_axis((0, kernel_h), name="kh")
    kw = te.reduce_axis((0, kernel_w), name="kw")
    c = te.reduce_axis((0, c), name="c")
    C = te.compute(
        out_shape,
        lambda n, h, w, f: te.sum(
            pad[n, h * stride_h + kh, w * stride_w + kw, c]
            * B[kh - 1 - tir.any(dilation_h), kw - 1 - tir.any(dilation_w), c, f],
            axis=[kh, kw, c],
        ),
        name="C",
    )
    return tvm.ir.IRModule({"main": te.create_prim_func([A, B, C])})


ir_module = conv2d_nhwc_hwnc(*(128, 64, 224, 224, 3, 7, 7, 2, 1, 3, "float16", "float16"))
func = ir_module["main"]

sch = tir.Schedule(func)
main_block = sch.get_block("C")
sch = normalize_with_tensorcore(sch, main_block)

print(sch.mod["main"])
