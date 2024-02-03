import tvm
from tvm.script import tir as T


@T.prim_func(private=True)
def func(var_inp0: T.handle, inp1: T.Buffer((4096, 4096), "float32"), var_matmul: T.handle):
    m = T.int32()
    inp0 = T.match_buffer(var_inp0, (1, m, 4096))
    matmul = T.match_buffer(var_matmul, (1, m, 4096))
    for i0, i1, i2, k in T.grid(1, m, 4096, 4096):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + inp0[v_i0, v_i1, v_k] * inp1[v_k, v_i2]
            
func = func

a, b, c = func.params

new = func.specialize({'m': 4096})

print(new)
