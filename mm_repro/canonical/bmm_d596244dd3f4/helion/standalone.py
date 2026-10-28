"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_d596244dd3f4
Shape hash: 844472c4
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([1024, 256, 256]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        tmp4 = tile_x.index.to(torch.int64)
        tmp5 = tile_y.index.to(torch.int64)
        tmp6 = tmp4.reshape([tile_x, 1]) <= tmp5.reshape([1, tile_y])
        tmp7 = hl.full([], 0.0, dtype=torch.float32)
        tmp8 = hl.full([], -3.3895313892515355e+38, dtype=torch.float32)
        tmp9 = torch.where(tmp6, tmp7, tmp8)
        tmp10 = acc + tmp9.permute(1, 0).reshape([1, tile_y, tile_x])
        tmp11 = tmp10.to(torch.float32)
        out0[tile_z, tile_y, tile_x] = tmp11.to(torch.float32)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, buf1)
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
    assert_size_stride_grouped((arg0_1, arg1_1), ((1024, 256, 64), (1024, 64, 256)), ((16384, 64, 1), (16384, 1, 64)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((1024, 256, 256), (65536, 256, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg0_1, arg1_1, buf1)
        del arg0_1
        del arg1_1
    return (buf1,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1024, 256, 64], 'dtype': 'bfloat16', 'stride': [16384, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024, 64, 256], 'dtype': 'bfloat16', 'stride': [16384, 1, 64], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [64, 16, 256, 256]}, {'kind': 'shape', 'name': None, 'dims': [64, -1, 256, 256]}, {'kind': 'shape', 'name': None, 'dims': [1024, 256, 256]}]}
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
