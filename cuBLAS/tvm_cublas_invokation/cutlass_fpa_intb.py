import numpy as np
import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relax
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
import time


def benchmark_fpa_intb():
    shapes = [
        (1, 1024, 8192),
        (1, 8192, 8192),
        (1, 8192, 28672),
        (1, 28672, 8192),
        
        # (1,	16384,	16384),
        # (1,	43008,	14336),
        # (1,	14336,	14336),
        # (1,	57344,	14336),
        # (1,	14336,	57344),
        # (1,	9216,	9216),
        # (1,	36864,	9216),
        # (1,	9216,	36864),
        # (1,	22016,	8192),
        # (1,	8192,	22016),
        # (1,	8192,	8192),
        # (1,	28672,	8192),
        # (1,	8192,	22016),
        # (16,	16384,	16384),
        # (16,	43008,	14336),
        # (16,	14336,	14336),
        # (16,	57344,	14336),
        # (16,	14336,	57344),
        # (16,	9216,	9216),
        # (16,	36864,	9216),
        # (16,	9216,	36864),
        # (16,	22016,	8192),
        # (16,	8192,	22016),
        # (16,	8192,	8192),
        # (16,	28672,	8192),
        # (16,	8192,	22016),
        # (32,	16384,	16384),
        # (32,	43008,	14336),
        # (32,	14336,	14336),
        # (32,	57344,	14336),
        # (32,	14336,	57344),
        # (32,	9216,	9216),
        # (32,	36864,	9216),
        # (32,	9216,	36864),
        # (32,	22016,	8192),
        # (32,	8192,	22016),
        # (32,	8192,	8192),
        # (32,	28672,	8192),
        # (32,	8192,	22016),
        # (64,	16384,	16384),
        # (64,	43008,	14336),
        # (64,	14336,	14336),
        # (64,	57344,	14336),
        # (64,	14336,	57344),
        # (64,	9216,	9216),
        # (64,	36864,	9216),
        # (64,	9216,	36864),
        # (64,	22016,	8192),
        # (64,	8192,	22016),
        # (64,	8192,	8192),
        # (64,	28672,	8192),
        # (64,	8192,	22016),
        # (128,	16384,	16384),
        # (128,	43008,	14336),
        # (128,	14336,	14336),
        # (128,	57344,	14336),
        # (128,	14336,	57344),
        # (128,	9216,	9216),
        # (128,	36864,	9216),
        # (128,	9216,	36864),
        # (128,	22016,	8192),
        # (128,	8192,	22016),
        # (128,	8192,	8192),
        # (128,	28672,	8192),
        # (128,	8192,	22016),
        # (256,	16384,	16384),
        # (256,	43008,	14336),
        # (256,	14336,	14336),
        # (256,	57344,	14336),
        # (256,	14336,	57344),
        # (256,	9216,	9216),
        # (256,	36864,	9216),
        # (256,	9216,	36864),
        # (256,	22016,	8192),
        # (256,	8192,	22016),
        # (256,	8192,	8192),
        # (256,	28672,	8192),
        # (256,	8192,	22016),
        # (512,	16384,	16384),
        # (512,	43008,	14336),
        # (512,	14336,	14336),
        # (512,	57344,	14336),
        # (512,	14336,	57344),
        # (512,	9216,	9216),
        # (512,	36864,	9216),
        # (512,	9216,	36864),
        # (512,	22016,	8192),
        # (512,	8192,	22016),
        # (512,	8192,	8192),
        # (512,	28672,	8192),
        # (512,	8192,	22016),
        # (1024,	16384,	16384),
        # (1024,	43008,	14336),
        # (1024,	14336,	14336),
        # (1024,	57344,	14336),
        # (1024,	14336,	57344),
        # (1024,	9216,	9216),
        # (1024,	36864,	9216),
        # (1024,	9216,	36864),
        # (1024,	22016,	8192),
        # (1024,	8192,	22016),
        # (1024,	8192,	8192),
        # (1024,	28672,	8192),
        # (1024,	8192,	22016),
        # (2048,	16384,	16384),
        # (2048,	43008,	14336),
        # (2048,	14336,	14336),
        # (2048,	57344,	14336),
        # (2048,	14336,	57344),
        # (2048,	9216,	9216),
        # (2048,	36864,	9216),
        # (2048,	9216,	36864),
        # (2048,	22016,	8192),
        # (2048,	8192,	22016),
        # (2048,	8192,	8192),
        # (2048,	28672,	8192),
        # (2048,	8192,	22016),
        # (4096,	16384,	16384),
        # (4096,	43008,	14336),
        # (4096,	14336,	14336),
        # (4096,	57344,	14336),
        # (4096,	14336,	57344),
        # (4096,	9216,	9216),
        # (4096,	36864,	9216),
        # (4096,	9216,	36864),
        # (4096,	22016,	8192),
        # (4096,	8192,	22016),
        # (4096,	8192,	8192),
        # (4096,	28672,	8192),
        # (4096,	8192,	22016),
        # (8192,	16384,	16384),
        # (8192,	43008,	14336),
        # (8192,	14336,	14336),
        # (8192,	57344,	14336),
        # (8192,	14336,	57344),
        # (8192,	9216,	9216),
        # (8192,	36864,	9216),
        # (8192,	9216,	36864),
        # (8192,	22016,	8192),
        # (8192,	8192,	22016),
        # (8192,	8192,	8192),
        # (8192,	28672,	8192),
        # (8192,	8192,	22016),
        # (16384,	16384,	16384),
        # (16384,	43008,	14336),
        # (16384,	14336,	14336),
        # (16384,	57344,	14336),
        # (16384,	14336,	57344),
        # (16384,	9216,	9216),
        # (16384,	36864,	9216),
        # (16384,	9216,	36864),
        # (16384,	22016,	8192),
        # (16384,	8192,	22016),
        # (16384,	8192,	8192),
        # (16384,	28672,	8192),
        # (16384,	8192,	22016),
    ]

    for M, N, K in shapes:

        @I.ir_module
        class Module:
            @T.prim_func
            def decode(
                A: T.Buffer((T.int64(K), T.int64(N // 2)), "int8"),
                B: T.Buffer((T.int64(N),), "float16"),
                decode_1: T.Buffer((T.int64(K), T.int64(N)), "float16"),
            ):
                T.func_attr({"tir.noalias": T.bool(True)})
                # with T.block("root"):
                for i, j in T.grid(T.int64(K), T.int64(N)):
                    with T.block("decode"):
                        v_i, v_j = T.axis.remap("SS", [i, j])
                        T.reads(A[v_i, v_j // T.int64(2)], B[v_j])
                        T.writes(decode_1[v_i, v_j])
                        decode_1[v_i, v_j] = (
                            T.Cast(
                                "float16",
                                T.shift_right(
                                    T.shift_left(
                                        T.bitwise_and(
                                            T.shift_right(
                                                T.Cast(
                                                    "int32", A[v_i, v_j // T.int64(2)]
                                                ),
                                                T.Cast("int32", v_j % T.int64(2)) * 4,
                                            ),
                                            15,
                                        ),
                                        28,
                                    ),
                                    28,
                                ),
                            )
                            * B[v_j]
                        )

            @R.function
            def main_bias(
                x: R.Tensor((M, K), dtype="float16"),
                y: R.Tensor((K, N // 2), dtype="int8"),
                scales: R.Tensor((N,), dtype="float16"),
            ) -> R.Tensor((M, N), dtype="float16"):
                R.func_attr({"num_input": 1})
                cls = Module
                with R.dataflow():
                    lv1 = y
                    lv2 = R.call_pure_packed(
                        "cutlass.ft_preprocess_weight_int4",
                        lv1,
                        80,
                        sinfo_args=(R.Tensor((K, N // 2), dtype="int8"),),
                    )
                    lv3: R.Tensor((N,), dtype="float16") = scales
                    lv6 = R.call_tir(
                        cls.decode,
                        (lv2, lv3),
                        out_sinfo=R.Tensor((K, N), dtype="float16"),
                    )
                    lv1_1: R.Tensor((M, N), dtype="float16") = R.matmul(
                        x, lv6, out_dtype="float16"
                    )
                    R.output(lv1_1)
                return lv1_1

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

        x_shape = (M, K)
        y_shape = (K, N // 2)

        mod = partition_for_cutlass(Module)
        func_names = [name.name_hint for (name, _) in mod.functions.items()]
        assert "fused_decode_relax_matmul_cutlass" in func_names

        mod = relax.transform.RunCodegen(
            {"cutlass": {"sm": 80, "find_first_valid": False}},
            entry_functions=["main_bias"],
        )(mod)

        x = np.random.randn(*x_shape).astype("float16")
        y = np.random.normal(0, 0.002, size=y_shape).astype("int8")

        mod = relax.pipeline.get_pipeline()(mod)
        mod = relax.transform.LiftTransformParams()(mod)

        _, mod_deploy, _ = split_transform_deploy_mod(mod)

        dev = tvm.device("cuda", 0)
        ex = relax.build(mod_deploy, target="cuda")
        vm = relax.vm.VirtualMachine(ex, dev)

        tvm_x = tvm.nd.array(x, dev)
        tvm_y = tvm.nd.array(y, dev)
        tvm_scales = tvm.nd.array(np.ones(N, dtype="float16"), dev)
        params = (tvm_y, tvm_scales)

        # warm up
        for i in range(5):
            vm["main_bias"](tvm_x, params)

        iters = 10
        dev.sync()
        start = time.time()
        for i in range(iters):
            vm["main_bias"](tvm_x, params)
        dev.sync()
        end = time.time()

        cost = (end - start) / iters * 1000
        print("cost: ", cost)


benchmark_fpa_intb()
