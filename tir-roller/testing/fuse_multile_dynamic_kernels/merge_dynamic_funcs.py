import numpy as np
import tvm
from tvm.script import tir as T
from tvm.tir.analysis import undefined_vars
from tvm.dlight.base.roller.policy import TensorCorePolicy, DefaultPolicy
from tvm.dlight.base.roller.arch import CUDA
from tvm.dlight.gpu import Matmul
from tvm.dlight.gpu.matmul_analysis import get_tensorized_func_and_tags
from tvm.dlight.base.utils import apply_and_build, apply_and_build_parallel
import time
from typing import Set, Dict, List

from scheduled_funcs import (
    original,
    matmul_opt_m_1024,
)

candidates = [original, matmul_opt_m_1024]


# always use the first function as the base
def collect_buffers_to_declare(func):
    params = []
    # collect dynamic symbolic
    dyn_symbolic: List[tvm.tir.Var] = []
    buffers_to_declare = []
    for param in func.params:
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        for axis in buffer.shape:
            if isinstance(axis, tvm.tir.Var) and axis not in dyn_symbolic:
                dyn_symbolic.append(axis)
        buffers_to_declare.append(buffer)
        params.append(buffer.data)

    # the args should be buffers + dynamic symbolic
    params += list(dyn_symbolic)

    return params, buffers_to_declare


params, buffers_to_declare = collect_buffers_to_declare(matmul_opt_m_1024)

def refactor_func(func, params, buffers_to_declare):
    body = func.body
    global_symbol = func.attrs["global_symbol"]
    if "opt_shapes" in func.attrs:
        opt_shapes = func.attrs["opt_shapes"]

    def serialize_name(opt_shapes: Dict):
        return "_opt_" + "_".join([f"{k}_{v}" for k, v in opt_shapes.items()])

    global_symbol += serialize_name(opt_shapes)
    ret_type = func.ret_type
    for buf in buffers_to_declare:
        body = tvm.tir.DeclBuffer(buf, body=body)

    device_func = tvm.tir.PrimFunc(params, body, ret_type).with_attrs(
        {"tir.is_global_func": True, "global_symbol": global_symbol}
    )
    return global_symbol, device_func


global_symbol, device_func = refactor_func(matmul_opt_m_1024, params, buffers_to_declare)
device_mod = tvm.IRModule.from_expr(device_func)
mod = tvm.IRModule()
mod.update(device_mod)

# print(device_mod)

def create_dispatch(func):
    body = func.body
    global_symbol = func.attrs["global_symbol"]
    attrs = func.attrs
    buffer_map = func.buffer_map
    params = func.params
    ret_type = func.ret_type

    # collect dynamic symbolic
    dyn_symbolic: List[tvm.tir.Var] = []
    _invoke_params = []
    for param in func.params:
        if param not in func.buffer_map:
            continue
        buffer = func.buffer_map[param]
        for axis in buffer.shape:
            if isinstance(axis, tvm.tir.Var) and axis not in dyn_symbolic:
                dyn_symbolic.append(axis)
        _invoke_params.append(buffer.data)
    _invoke_params += list(dyn_symbolic)

    ib = tvm.tir.ir_builder.create()
    m = list(dyn_symbolic)[-1]
    with ib.if_scope(m <= 1024):
        ib.emit(tvm.tir.call_packed(global_symbol, *_invoke_params))
    stmt = ib.get()
    dispatch_func = tvm.tir.PrimFunc(params, stmt, ret_type, buffer_map, attrs).with_attrs(
        {"tir.is_global_func": True, "global_symbol": global_symbol}
    )
    return dispatch_func

dispatch_func = create_dispatch(original)
print(dispatch_func)
