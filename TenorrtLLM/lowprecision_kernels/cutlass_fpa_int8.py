import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relax
from tvm.contrib.cutlass.build import is_shape_valid_for_cutlass_matmul
from tvm.contrib.pickle_memoize import memoize
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.testing import get_relax_matmul_module
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import relax as relax_builder
import time

VERIFY = True
shapes = [
    # M, N, K
    # (1, 16384, 16384),
    (1, 12800, 4096),
    # (1, 4096, 6400),
    # (16, 16384, 16384),
    # (16, 12800, 4096),
    # (16, 4096, 6400),
    # (32, 16384, 16384),
    # (32, 12800, 4096),
    # (32, 4096, 6400),
    # (64, 16384, 16384),
    # (64, 12800, 4096),
    # (64, 4096, 6400),
    # (128, 16384, 16384),
    # (128, 12800, 4096),
    # (128, 4096, 6400),
    # (256, 16384, 16384),
    # (256, 12800, 4096),
    # (256, 4096, 6400),
]

perf_map = []
def split_transform_deploy_mod(mod):
    mod_transform = tvm.IRModule()
    mod_deploy = tvm.IRModule().with_attrs(mod.attrs)

    transform_func_name = None

    for gv, func in mod.functions.items():
        if "transform_params" in gv.name_hint:
            transform_func_name = gv.name_hint
            mod_transform[gv] = func
        elif isinstance(func, tvm.tir.PrimFunc):
            mod_transform[gv] = func
        else:
            mod_deploy[gv] = func

    assert transform_func_name is not None
    return mod_transform, mod_deploy, transform_func_name


for M, N, K in shapes:
    @I.ir_module
    class Module:
        @T.prim_func
        def decode(
            A: T.Buffer((T.int64(K), T.int64(N)), "int8"),
            B: T.Buffer((T.int64(N),), "float16"),
            decode_1: T.Buffer((T.int64(K), T.int64(N)), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            for i, j in T.grid(T.int64(K), T.int64(N)):
                with T.block("decode"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(A[v_i, v_j], B[v_j])
                    T.writes(decode_1[v_i, v_j])
                    decode_1[v_i, v_j] = T.Cast("float16", A[v_i, v_j]) * B[v_j]

        @T.prim_func
        def encode(
            A: T.Buffer((T.int64(N), T.int64(K)), "float16"),
            w_gathered: T.Buffer((T.int64(K), T.int64(N)), "int8"),
            compute: T.Buffer((T.int64(N),), "float16"),
        ):
            T.func_attr({"tir.noalias": T.bool(True)})
            # with T.block("root"):
            max_abs_value = T.alloc_buffer((T.int64(N),), "float16")
            scale = T.alloc_buffer((T.int64(N),))
            for i, k in T.grid(T.int64(N), T.int64(K)):
                with T.block("max_abs_value"):
                    v_i, v_k = T.axis.remap("SR", [i, k])
                    T.reads(A[v_i, v_k])
                    T.writes(max_abs_value[v_i])
                    with T.init():
                        max_abs_value[v_i] = T.float16(-65504)
                    max_abs_value[v_i] = T.max(max_abs_value[v_i], T.fabs(A[v_i, v_k]))
            for i in range(T.int64(N)):
                with T.block("scale"):
                    v_i = T.axis.spatial(T.int64(N), i)
                    T.reads(max_abs_value[v_i])
                    T.writes(scale[v_i])
                    scale[v_i] = T.max(
                        T.Cast("float32", max_abs_value[v_i]), T.float32(0.0001)
                    ) * T.float32(0.0078125)
            for j, i in T.grid(T.int64(K), T.int64(N)):
                with T.block("w_gathered"):
                    v_j, v_i = T.axis.remap("SS", [j, i])
                    T.reads(A[v_i, v_j], scale[v_i])
                    T.writes(w_gathered[v_j, v_i])
                    w_gathered[v_j, v_i] = T.Cast(
                        "int8",
                        T.min(
                            T.max(
                                T.round(T.Cast("float32", A[v_i, v_j]) / scale[v_i]),
                                T.float32(-128),
                            ),
                            T.float32(127),
                        ),
                    )
            for i0 in range(T.int64(N)):
                with T.block("compute"):
                    v_i0 = T.axis.spatial(T.int64(N), i0)
                    T.reads(scale[v_i0])
                    T.writes(compute[v_i0])
                    compute[v_i0] = T.Cast("float16", scale[v_i0])

        @R.function
        def main(
            x: R.Tensor((M, K), dtype="float16"),
            y: R.Tensor((N, K), dtype="float16")
        ) -> R.Tensor((M, N), dtype="float16"):
            R.func_attr({"num_input": 1})
            cls = Module
            with R.dataflow():
                lv = R.call_tir(
                    cls.encode,
                    (y,),
                    out_sinfo=[R.Tensor((K, N), dtype="int8"), R.Tensor((N,), dtype="float16")],
                )
                lv1 = lv[0]
                lv2 = R.call_pure_packed(
                    "cutlass.ft_preprocess_weight",
                    lv1,
                    R.prim_value(80),
                    R.prim_value(0),
                    sinfo_args=(R.Tensor((K, N), dtype="int8"),),
                )
                lv3 = lv[1]
                lv4 = R.builtin.stop_lift_params(lv2)
                lv5 = R.builtin.stop_lift_params(lv3)
                lv6 = R.call_tir(
                    cls.decode, (lv4, lv5), out_sinfo=R.Tensor((K, N), dtype="float16")
                )
                lv1_1= R.matmul(x, lv6, out_dtype="float16")
                R.output(lv1_1)
            return lv1_1
    x_shape = (M, K)
    y_shape = (N, K)
    mod = partition_for_cutlass(Module)
    func_names = [name.name_hint for (name, _) in mod.functions.items()]

    mod = relax.transform.RunCodegen(
        {"cutlass": {"sm": 80, "find_first_valid": False}},
        entry_functions=["main"],
    )(mod)

    mod = relax.pipeline.get_pipeline()(mod)
    mod = relax.transform.LiftTransformParams()(mod)
    mod_transform, mod_deploy, transform_function_name = split_transform_deploy_mod(mod)

    print(mod_deploy)
    dev = tvm.device("cuda", 0)
    ex = relax.build(mod_deploy, target="nvidia/nvidia-a100")
    vm = relax.vm.VirtualMachine(ex, dev)
    numpy_a = np.random.uniform(0, 1, [M, K]).astype("float16")
    numpy_b = np.random.randint(-127, 128, [K, N]).astype("int8")
    tvm_a = tvm.nd.array(numpy_a, device=dev)
    
    tvm_b = tvm.nd.array(
        numpy_b, device=dev
    )
    tvm_scales = tvm.nd.array(np.ones(N, dtype="float16"), dev)
    vm_output = vm["main"](tvm_a, tvm_b, tvm_scales)
    print("vm_output is ", vm_output)

    if VERIFY:
        ref_output = np.matmul(numpy_a, numpy_b.astype("float16"))
        print("ref_output is ", ref_output)
        ex = relax.build(mod_transform, target="llvm")
        vm = relax.vm.VirtualMachine(ex, tvm.cpu(0))
        x_shape = (M, K)
        y_shape = (N, K)
        x = np.random.randn(*x_shape).astype("float16")
        y = np.random.normal(0, 0.002, size=y_shape).astype("float16")
        packed_weight, scales = vm[transform_function_name](
            (tvm.nd.array(y), )
        )
        dev = tvm.device("cuda", 0)
        ex = relax.build(mod_deploy, target="cuda")
        vm = relax.vm.VirtualMachine(ex, dev)

        x_nd = tvm.nd.array(x, dev)
        inp = [x_nd, packed_weight.copyto(dev), scales.copyto(dev)]
        out = vm["main"](*inp).numpy()
        ref = np.dot(x, y.transpose())
        print("out is ", out)
    # warm up
    for i in range(5):
        vm["main"](tvm_a, tvm_b, tvm_scales)

    iters = 10
    dev.sync()
    start = time.time()
    for i in range(iters):
        vm["main"](tvm_a, tvm_b, tvm_scales)
    dev.sync()
    end = time.time()

    cost = (end - start) / iters * 1000
    print("cost: ", cost)
    key = f"{M}_{N}_{K}"
    perf_map.append((key, cost))

for key, cost in perf_map:
    print(f"{key}: {cost}")
