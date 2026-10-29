"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_9eb920d9a271
Shape hash: 7bd269be
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1_op2(A: torch.Tensor, B: torch.Tensor, A1: torch.Tensor, B1: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 1000]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = A1[tile_y, tile_r].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            tmp6 = B1[tile_r, tile_x].to(torch.float32)
            tmp7 = tmp6.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp5, tmp7)
        tmp8 = in0[tile_x].to(torch.float32)
        tmp9 = hl.full([], 1.0, dtype=torch.float32)
        tmp10 = tmp9 * tmp8
        tmp11 = tmp9 * acc
        tmp12 = tmp10 + tmp11
        tmp13 = in1[tile_x].to(torch.float32)
        tmp14 = tmp9 * tmp13
        tmp15 = tmp9 * acc1
        tmp16 = tmp14 + tmp15
        tmp17 = tmp12 + tmp16
        tmp18 = hl.full([], 0.5, dtype=torch.float32)
        tmp19 = tmp17 * tmp18
        out0[tile_y, tile_x] = tmp19.to(torch.bfloat16)

def call_helion_op0_op1_op2(arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg4_1: torch.Tensor, arg5_1: torch.Tensor, arg0_1: torch.Tensor, arg3_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op0_op1_op2(arg1_1, arg2_1, arg4_1, arg5_1, arg0_1, arg3_1, buf2)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride_grouped((arg1_1, arg2_1, arg4_1, arg5_1, arg0_1, arg3_1), ((128, 768), (768, 1000), (128, 768), (768, 1000), (1000,), (1000,)), ((152064, 1), (1, 768), (152064, 1), (1, 768), (1,), (1,)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf0 = empty_strided_cuda((128, 1000), (1000, 1), torch.bfloat16)
        buf2 = buf0
        del buf0
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1_op2(arg1_1, arg2_1, arg4_1, arg5_1, arg0_1, arg3_1, buf2)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
    return (buf2,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [768, 1000], 'dtype': 'bfloat16', 'stride': [1, 768], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 768], 'dtype': 'bfloat16', 'stride': [152064, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 1000], 'dtype': 'bfloat16', 'stride': [1, 768], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 768], 'dtype': 'bfloat16', 'stride': [152064, 1], 'device': 'cuda', 'storage_offset': 768, 'alias_group': 0}], 'alias_group_nbytes': [38928384]}
COMPILED_INPUT_ORDER = [1, 2, 0, 4, 5, 3]

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
