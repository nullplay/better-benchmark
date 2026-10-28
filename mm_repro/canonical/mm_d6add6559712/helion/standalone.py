"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_d6add6559712
Shape hash: 771bbc6c
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([256, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_y, tile_x].to(torch.float32)
        tmp5 = tmp4.to(torch.float32)
        tmp6 = hl.full([], 0.0, dtype=torch.float32)
        tmp7 = tmp5 <= tmp6
        tmp8 = in1[tile_y, tile_x].to(torch.bool)
        tmp9 = tmp8.to(torch.float32)
        tmp10 = hl.full([], 5.000000000000001, dtype=torch.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = acc * tmp11
        tmp13 = tmp12.to(torch.float32)
        tmp14 = hl.full([], 1.7580993408473766, dtype=torch.float32)
        tmp15 = tmp13 * tmp14
        tmp16 = torch.exp(tmp5)
        tmp17 = tmp15 * tmp16
        tmp18 = hl.full([], 1.0507009873554805, dtype=torch.float32)
        tmp19 = tmp13 * tmp18
        tmp20 = torch.where(tmp7, tmp17, tmp19)
        tmp21 = tmp20.to(torch.float32)
        out0[tile_y, tile_x] = tmp21.to(torch.bfloat16)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg3_1: torch.Tensor, arg2_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, arg3_1, arg2_1, buf1)
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
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg3_1, arg2_1), ((256, 512), (512, 1024), (256, 1024), (256, 1024)), ((512, 1), (1024, 1), (1024, 1), (1024, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf0 = empty_strided_cuda((256, 1024), (1024, 1), torch.bfloat16)
        buf1 = buf0
        del buf0
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg0_1, arg1_1, arg3_1, arg2_1, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
    return (buf1, reinterpret_tensor(buf1, (1024, 256), (1, 1024), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [256, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [256, 1024], 'dtype': 'bool', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [256, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3]

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
