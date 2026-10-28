"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_079dfeb381b2
Shape hash: 41c7b013
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([384, 512, 512]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        tmp4 = hl.full([], 0.125, dtype=torch.float32)
        tmp5 = acc * tmp4
        tmp6 = tmp5.to(torch.float32)
        out0[tile_z.index.reshape([tile_z, 1, 1]) // 12, tile_z.index.reshape([tile_z, 1, 1]) % 12, tile_y.index.reshape([1, tile_y, 1]), tile_x.index.reshape([1, 1, tile_x])] = tmp6.to(torch.float32)
        tmp7 = tmp6 == tmp6
        tmp8 = torch.abs(tmp6)
        tmp9 = hl.full([], float('inf'), dtype=torch.float32)
        tmp10 = tmp8 != tmp9
        tmp11 = torch.logical_and(tmp7, tmp10)
        tmp12 = torch.logical_not(tmp11)
        out1[tile_z.index.reshape([tile_z, 1, 1]) // 12, tile_z.index.reshape([tile_z, 1, 1]) % 12, tile_y.index.reshape([1, tile_y, 1]), tile_x.index.reshape([1, 1, tile_x])] = tmp12.to(torch.bool)

def call_helion_op0_op1_op2(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op0_op1_op2(arg0_1, arg1_1, buf1, buf2)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
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
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((384, 512, 64), (384, 64, 512)), ((32768, 64, 1), (32768, 512, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((32, 12, 512, 512), (3145728, 262144, 512, 1), torch.float32)
        buf2 = empty_strided_cuda((32, 12, 512, 512), (3145728, 262144, 512, 1), torch.bool)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1_op2(arg0_1, arg1_1, buf1, buf2)
        del arg0_1
        del arg1_1
    return (buf1, buf2)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [384, 512, 64], 'dtype': 'bfloat16', 'stride': [32768, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 64, 512], 'dtype': 'bfloat16', 'stride': [32768, 512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 12, 512, 512]}]}
COMPILED_INPUT_ORDER = [0, 1]

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
