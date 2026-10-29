"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_0ebedcb77817
Shape hash: 289209ba
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_y, tile_x].to(torch.bool)
        tmp5 = tmp4.to(torch.float32)
        tmp6 = hl.full([], 1.1111111111111112, dtype=torch.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = acc * tmp7
        tmp9 = tmp8.to(torch.float32)
        tmp10 = in1[tile_y, tile_x].to(torch.float32)
        tmp11 = tmp10.to(torch.float32)
        tmp12 = hl.full([], 0.7071067811865476, dtype=torch.float32)
        tmp13 = tmp11 * tmp12
        tmp14 = torch.erf(tmp13)
        tmp15 = hl.full([], 1.0, dtype=torch.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = hl.full([], 0.5, dtype=torch.float32)
        tmp18 = tmp16 * tmp17
        tmp19 = tmp11 * tmp11
        tmp20 = hl.full([], -0.5, dtype=torch.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = torch.exp(tmp21)
        tmp23 = hl.full([], 0.3989422804014327, dtype=torch.float32)
        tmp24 = tmp22 * tmp23
        tmp25 = tmp11 * tmp24
        tmp26 = tmp18 + tmp25
        tmp27 = tmp9 * tmp26
        tmp28 = tmp27.to(torch.float32)
        out0[tile_y, tile_x] = tmp28.to(torch.bfloat16)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg3_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, reinterpret_tensor(arg2_1, (8192, 4096), (4096, 1), 0), arg3_1, reinterpret_tensor(buf1, (8192, 4096), (4096, 1), 0))
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
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1, arg3_1), ((8192, 1024), (1024, 4096), (512, 16, 4096), (8192, 4096)), ((1024, 1), (4096, 1), (65536, 4096, 1), (4096, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf0 = empty_strided_cuda((8192, 4096), (4096, 1), torch.bfloat16)
        buf1 = reinterpret_tensor(buf0, (512, 16, 4096), (65536, 4096, 1), 0)
        del buf0
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg0_1, arg1_1, arg2_1, arg3_1, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
    return (reinterpret_tensor(buf1, (8192, 4096), (4096, 1), 0), reinterpret_tensor(buf1, (4096, 8192), (1, 4096), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [8192, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 16, 4096], 'dtype': 'bool', 'stride': [65536, 4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 4096]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 4096]}, {'kind': 'shape', 'name': None, 'dims': [8192, 4096]}]}
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
