"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_bc1681dbe7a5
Shape hash: 2fcaeafc
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1_op2_op3_op4(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = in0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.bool)
        tmp6 = tmp5.to(torch.float32)
        tmp7 = hl.full([], 1.1111111111111112, dtype=torch.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp4 * tmp8
        tmp10 = in1[tile_y, tile_x].to(torch.float32)
        tmp11 = hl.full([], 0.5, dtype=torch.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp12.to(torch.float32)
        tmp14 = tmp10.to(torch.float32)
        tmp15 = tmp14 * tmp14
        tmp16 = tmp15 * tmp14
        tmp17 = hl.full([], 0.044715, dtype=torch.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp14 + tmp18
        tmp20 = hl.full([], 0.7978845608028654, dtype=torch.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = torch.tanh(tmp21)
        tmp23 = hl.full([], 1.0, dtype=torch.float32)
        tmp24 = tmp22 + tmp23
        tmp25 = tmp13 * tmp24
        tmp26 = tmp9 * tmp25
        tmp27 = tmp26.to(torch.float32)
        out0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp27.to(torch.bfloat16)
        tmp28 = acc.to(torch.float32)
        tmp29 = in0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.bool)
        tmp30 = tmp29.to(torch.float32)
        tmp31 = hl.full([], 1.1111111111111112, dtype=torch.float32)
        tmp32 = tmp30 * tmp31
        tmp33 = tmp28 * tmp32
        tmp34 = in2[tile_y, tile_x].to(torch.float32)
        tmp35 = tmp34.to(torch.float32)
        tmp36 = tmp33 * tmp35
        tmp37 = in1[tile_y, tile_x].to(torch.float32)
        tmp38 = hl.full([], 0.5, dtype=torch.float32)
        tmp39 = tmp37 * tmp38
        tmp40 = tmp39.to(torch.float32)
        tmp41 = tmp36 * tmp40
        tmp42 = tmp37.to(torch.float32)
        tmp43 = tmp42 * tmp42
        tmp44 = tmp43 * tmp42
        tmp45 = hl.full([], 0.044715, dtype=torch.float32)
        tmp46 = tmp44 * tmp45
        tmp47 = tmp42 + tmp46
        tmp48 = hl.full([], 0.7978845608028654, dtype=torch.float32)
        tmp49 = tmp47 * tmp48
        tmp50 = torch.tanh(tmp49)
        tmp51 = tmp50 * tmp50
        tmp52 = hl.full([], 1.0, dtype=torch.float32)
        tmp53 = tmp52 - tmp51
        tmp54 = tmp41 * tmp53
        tmp55 = tmp54 * tmp48
        tmp56 = tmp55 * tmp45
        tmp57 = hl.full([], 3.0, dtype=torch.float32)
        tmp58 = tmp43 * tmp57
        tmp59 = tmp56 * tmp58
        tmp60 = tmp55.to(torch.float32)
        tmp61 = tmp59.to(torch.float32)
        tmp62 = tmp60 + tmp61
        tmp63 = tmp50 + tmp52
        tmp64 = tmp36 * tmp63
        tmp65 = tmp64.to(torch.float32)
        tmp66 = tmp65 * tmp38
        tmp67 = tmp62 + tmp66
        out1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp67.to(torch.bfloat16)

def call_helion_op0_op1_op2_op3_op4(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg3_1: torch.Tensor, arg4_1: torch.Tensor, buf1: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op0_op1_op2_op3_op4(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf1, buf4)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
async_compile = AsyncCompile()
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1, arg3_1, arg4_1), ((4096, 512), (512, 1024), (32, 128, 1024), (4096, 1024), (4096, 1024)), ((512, 1), (1024, 1), (131072, 1024, 1), (1024, 1), (1024, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf1 = empty_strided_cuda((32, 128, 1024), (131072, 1024, 1), torch.bfloat16)
        buf3 = empty_strided_cuda((32, 128, 1024), (131072, 1024, 1), torch.bfloat16)
        buf4 = buf3
        del buf3
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1_op2_op3_op4(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf1, buf4)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
    return (reinterpret_tensor(buf1, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf1, (1024, 4096), (1, 1024), 0), reinterpret_tensor(buf4, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf4, (1024, 4096), (1, 1024), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 1024], 'dtype': 'bool', 'stride': [131072, 1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 1024]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 1024]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 1024]}, {'kind': 'shape', 'name': None, 'dims': [4096, 1024]}, {'kind': 'shape', 'name': None, 'dims': [4096, 1024]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4]

def get_args():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from repro_harness import make_inputs_from_config
    original = make_inputs_from_config(FIRST_SHAPE_CONFIG)
    return [original[i] for i in COMPILED_INPUT_ORDER]
make_inputs = get_args
if __name__ == '__main__':
    benchmark_compiled_module(get_args())
