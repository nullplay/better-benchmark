"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_059e5a3b520a
Shape hash: 4239d79a
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, in3: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 16384]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = in0[tile_y, tile_x].to(torch.float32)
        tmp6 = tmp5.to(torch.float32)
        tmp7 = tmp4 * tmp6
        tmp8 = in1[tile_y, tile_x].to(torch.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = hl.full([], 0.7978845608028654, dtype=torch.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11.to(torch.float32)
        tmp13 = hl.full([], 0.044715, dtype=torch.float32)
        tmp14 = tmp11 * tmp13
        tmp15 = in2[tile_y, tile_x].to(torch.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16.to(torch.float32)
        tmp18 = tmp12 + tmp17
        tmp19 = in3[tile_y, tile_x].to(torch.float32)
        tmp20 = tmp4 * tmp19
        tmp21 = tmp20.to(torch.float32)
        tmp22 = hl.full([], 0.5, dtype=torch.float32)
        tmp23 = tmp21 * tmp22
        tmp24 = tmp18 + tmp23
        out0[tile_y, tile_x] = tmp24.to(torch.bfloat16)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg4_1: torch.Tensor, arg5_1: torch.Tensor, arg3_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, reinterpret_tensor(arg2_1, (4096, 16384), (16384, 1), 0), reinterpret_tensor(arg4_1, (4096, 16384), (16384, 1), 0), reinterpret_tensor(arg5_1, (4096, 16384), (16384, 1), 0), reinterpret_tensor(arg3_1, (4096, 16384), (16384, 1), 0), reinterpret_tensor(buf1, (4096, 16384), (16384, 1), 0))
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1, arg4_1, arg5_1, arg3_1), ((4096, 4096), (4096, 16384), (8, 512, 16384), (8, 512, 16384), (8, 512, 16384), (8, 512, 16384)), ((4096, 1), (16384, 1), (8388608, 16384, 1), (8388608, 16384, 1), (8388608, 16384, 1), (8388608, 16384, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf0 = empty_strided_cuda((4096, 16384), (16384, 1), torch.bfloat16)
        buf1 = reinterpret_tensor(buf0, (8, 512, 16384), (8388608, 16384, 1), 0)
        del buf0
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg0_1, arg1_1, arg2_1, arg4_1, arg5_1, arg3_1, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
    return (reinterpret_tensor(buf1, (4096, 16384), (16384, 1), 0), reinterpret_tensor(buf1, (16384, 4096), (1, 16384), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 16384], 'dtype': 'bfloat16', 'stride': [16384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 512, 16384], 'dtype': 'bfloat16', 'stride': [8388608, 16384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 512, 16384], 'dtype': 'float32', 'stride': [8388608, 16384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 512, 16384], 'dtype': 'float32', 'stride': [8388608, 16384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 512, 16384], 'dtype': 'float32', 'stride': [8388608, 16384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 16384]}, {'kind': 'shape', 'name': None, 'dims': [4096, 16384]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5]

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
