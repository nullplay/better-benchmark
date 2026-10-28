"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_aaad50da7ef4
Shape hash: 4bf551a0
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([64, 1024, 1024]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        tmp4 = (-1 * min(0, tile_x.index[None, :] + -1 * tile_y.index[:, None])).to(torch.int64)
        tmp5 = hl.full([], 16, dtype=torch.int64)
        tmp6 = tmp4 < tmp5
        tmp7 = (-1 * min(0, tile_x.index[None, :] + -1 * tile_y.index[:, None])).to(torch.float32)
        tmp8 = hl.full([], 0.0625, dtype=torch.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = torch.log(tmp9)
        tmp11 = hl.full([], 0.48089834696298783, dtype=torch.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = hl.full([], 16.0, dtype=torch.float32)
        tmp14 = tmp12 * tmp13
        tmp15 = tmp14.to(torch.int64)
        tmp16 = tmp15 + tmp5
        tmp17 = hl.full([], 31, dtype=torch.int64)
        tmp18 = torch.minimum(tmp16, tmp17)
        tmp19 = torch.where(tmp6, tmp4, tmp18)
        tmp20 = in0[tmp19.reshape([1, tile_y, tile_x]), tile_z.index.reshape([tile_z, 1, 1]) % 8].to(torch.float32)
        tmp21 = tile_x.index.to(torch.int64)
        tmp22 = tile_y.index.to(torch.int64)
        tmp23 = tmp21.reshape([tile_x, 1]) <= tmp22.reshape([1, tile_y])
        tmp24 = hl.full([], 0.0, dtype=torch.float32)
        tmp25 = hl.full([], -3.3895313892515355e+38, dtype=torch.float32)
        tmp26 = torch.where(tmp23, tmp24, tmp25)
        tmp27 = tmp20 + tmp26.permute(1, 0).reshape([1, tile_y, tile_x])
        tmp28 = acc + tmp27
        tmp29 = tmp28.to(torch.float32)
        out0[tile_z.index.reshape([tile_z, 1, 1]) // 8, tile_z.index.reshape([tile_z, 1, 1]) % 8, tile_y.index.reshape([1, tile_y, 1]), tile_x.index.reshape([1, 1, tile_x])] = tmp29.to(torch.float32)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, arg2_1, buf1)
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
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1), ((64, 1024, 64), (64, 64, 1024), (32, 8)), ((65536, 64, 1), (65536, 1024, 1), (8, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((8, 8, 1024, 1024), (8388608, 1048576, 1024, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg0_1, arg1_1, arg2_1, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf1,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [64, 1024, 64], 'dtype': 'bfloat16', 'stride': [65536, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [64, 64, 1024], 'dtype': 'bfloat16', 'stride': [65536, 1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 8], 'dtype': 'bfloat16', 'stride': [8, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8, 8, 1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [8, -1, 1024, 1024]}]}
COMPILED_INPUT_ORDER = [0, 1, 2]

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
