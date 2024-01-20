import datetime
import os
import csv
import json
import argparse
import logging
from typing import Dict
import numpy as np  # type: ignore
import time
import tvm
from tvm import relay, relax, runtime, transform
from tvm.ir.module import IRModule
from tvm.relax.testing import relay_translator, nn
from tvm.target.target import Target

import os
from tvm import dlight as dl

fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
log_path = "progress/" + fname

count = 0


def write_code(code, path, fname):
    global count
    fname = str(count) + "." + fname
    count += 1
    if not os.path.exists(path):
        os.makedirs(path)
    fname = os.path.join(path, fname)
    with open(fname, "w") as f:
        f.write(code)


def write_sch(sch, path, fname):
    py_fname = fname + ".py"
    write_code(sch.mod["main"].script(), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(sch.mod.astext(), path, cu_fname)


def write_mod(mod, path, fname):
    py_fname = fname + ".py"
    write_code(mod.script(show_meta=False), path, py_fname)
    cu_fname = fname + ".cu"
    write_code(mod.astext(show_meta_data=False), path, cu_fname)


def apply_opt_before_tuning(
    relay_mod: IRModule, params: Dict[str, runtime.NDArray], target: Target
):
    with transform.PassContext(opt_level=3):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        write_mod(relay_mod, log_path, "create_mod")
        relay_mod = relay.transform.SimplifyInference()(relay_mod)
        write_mod(relay_mod, log_path, "SimplifyInference")
        relay_mod = relay.transform.ConvertLayout({"nn.conv2d": ["NHWC", "default"]})(relay_mod)
        write_mod(relay_mod, log_path, "ConvertLayout")
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        write_mod(relay_mod, log_path, "FoldConstant")
        relay_mod = relay.transform.FoldScaleAxis()(relay_mod)
        write_mod(relay_mod, log_path, "FoldScaleAxis")
        relay_mod = relay.transform.CanonicalizeOps()(relay_mod)
        write_mod(relay_mod, log_path, "CanonicalizeOps")
        relay_mod = relay.transform.AlterOpLayout()(relay_mod)
        write_mod(relay_mod, log_path, "AlterOpLayout")
        relay_mod = relay.transform.FoldConstant()(relay_mod)
        write_mod(relay_mod, log_path, "FoldConstant")

        relax_mod = relay_translator.from_relay(relay_mod["main"], target=target)
        write_mod(relax_mod, log_path, "relay_translator_relax")
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        write_mod(relax_mod, log_path, "AnnotateTIROpPattern")
        relax_mod = relax.transform.FuseOps()(relax_mod)
        write_mod(relax_mod, log_path, "FuseOps")
        relax_mod = relax.transform.FuseTIR()(relax_mod)
        write_mod(relax_mod, log_path, "FuseTIR")
    return relax_mod


# build conv relay model
n, h, w, ic, oc, kh, kw, stride, padding = 128, 224, 224, 3, 64, 7, 7, 2, 3

dtype = "float16"
target = tvm.target.Target("cuda")
# conv2d
conv2d = relay.nn.conv2d
A = relay.var("A", shape=(n, h, w, ic), dtype=dtype)
W = relay.var("W", shape=(kh, kw, ic, oc), dtype=dtype)
Conv = conv2d(
    A,
    W,
    strides=(stride, stride),
    padding=(padding, padding),
    kernel_size=(kh, kw),
    data_layout="NHWC",
    kernel_layout="HWIO",
)

f = relay.Function([A, W], Conv)
relay_mod = tvm.IRModule.from_expr(f)
relax_mod = apply_opt_before_tuning(relay_mod, {}, target)

from copy import deepcopy

ref_mod = deepcopy(relax_mod)


with target:
    relax_mod = dl.ApplyFastTuning(topk=20)(relax_mod)
# run codegen
write_mod(relax_mod, log_path, "ApplyFastTuning")

relax_mod = relax.transform.RunCodegen()(relax_mod)

write_mod(relax_mod, log_path, "run_codegen")

relax_mod = tvm.tir.transform.MakePackedAPI()(relax_mod)
write_mod(relax_mod, log_path, "make_packed_api")

ex = relax.build(relax_mod, target)
write_code(ex.mod.imported_modules[0].imported_modules[0].get_source(), log_path, "tmp.cu")

device = tvm.cuda(0)
vm = relax.VirtualMachine(ex, device)

# init parameters
params = nn.init_params(relax_mod)

A = np.random.uniform(size=(n, h, w, ic)).astype(dtype)
W = np.random.uniform(size=(kh, kw, ic, oc)).astype(dtype)

tvm_a = tvm.nd.array(A, device)
tvm_b = tvm.nd.array(W, device)

res = vm["main"](tvm_a, tvm_b)

device.sync()

start = time.time()

for i in range(10):
    vm["main"](tvm_a, tvm_b)


device.sync()

end = time.time()

print("Time cost is: ", (end - start) * 100, "ms")


def get_ref_output(ref_mod):
    write_mod(ref_mod, log_path, "the_ref_mod")

    with target:
        ref_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(ref_mod)
        ref_mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(ref_mod)
        ref_mod = dl.ApplyDefaultSchedule(dl.gpu.Reduction())(ref_mod)
        ref_mod = dl.ApplyDefaultSchedule(dl.gpu.GeneralReduction())(ref_mod)
        ref_mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(ref_mod)

    # run codegen
    write_mod(ref_mod, log_path, "apply_default_schedule")

    ref_mod = relax.transform.RunCodegen()(ref_mod)

    write_mod(ref_mod, log_path, "run_codegen")

    ref_mod = tvm.tir.transform.MakePackedAPI()(ref_mod)
    write_mod(ref_mod, log_path, "make_packed_api")

    ex = relax.build(ref_mod, target)
    write_code(ex.mod.imported_modules[0].imported_modules[0].get_source(), log_path, "tmp.cu")

    device = tvm.cuda(0)
    vm = relax.VirtualMachine(ex, device)
    res = vm["main"](tvm_a, tvm_b)


get_ref_output(ref_mod)
