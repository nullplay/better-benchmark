"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_8e49d8639fb2
Shape hash: 72bac7d3
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1024, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(8192):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp4.to(torch.float32)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, buf1)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2_op4_op6_op3_op5_op7(A: torch.Tensor, B: torch.Tensor, A1: torch.Tensor, A2: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1024, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc2 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(8192):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = A1[tile_y, tile_r].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp5, tmp3)
            tmp6 = A2[tile_y, tile_r].to(torch.float32)
            tmp7 = tmp6.to(torch.bfloat16)
            acc2 = torch.addmm(acc2, tmp7, tmp3)
        tmp8 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp8.to(torch.float32)
        tmp9 = acc1.to(torch.float32)
        out1[tile_y, tile_x] = tmp9.to(torch.float32)
        tmp10 = acc2.to(torch.float32)
        out2[tile_y, tile_x] = tmp10.to(torch.float32)

def call_helion_op2_op4_op6_op3_op5_op7(arg4_1: torch.Tensor, arg5_1: torch.Tensor, arg3_1: torch.Tensor, arg2_1: torch.Tensor, buf3: torch.Tensor, buf5: torch.Tensor, buf7: torch.Tensor) -> None:
    helion_op2_op4_op6_op3_op5_op7(reinterpret_tensor(arg4_1, (1024, 8192), (1, 1024), 0), arg5_1, reinterpret_tensor(arg3_1, (1024, 8192), (1, 1024), 0), reinterpret_tensor(arg2_1, (1024, 8192), (1, 1024), 0), buf3, buf5, buf7)
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
    assert_size_stride_grouped((arg0_1, arg1_1), ((1024, 8192), (8192, 1024)), ((1, 1024), (1024, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg0_1, arg1_1, buf1)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg4_1, arg5_1, arg3_1, arg2_1), ((16, 16, 512, 64), (8192, 1024), (16, 16, 512, 64), (16, 16, 512, 64)), ((524288, 64, 1024, 1), (1024, 1), (524288, 64, 1024, 1), (524288, 64, 1024, 1)), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf3 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        buf5 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        buf7 = empty_strided_cuda((1024, 1024), (1024, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2_op4_op6_op3_op5_op7(arg4_1, arg5_1, arg3_1, arg2_1, buf3, buf5, buf7)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
    return (buf1, buf3, buf5, buf7)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1024, 8192], 'dtype': 'bfloat16', 'stride': [1, 1024], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 16, 512, 64], 'dtype': 'bfloat16', 'stride': [524288, 64, 1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 16, 512, 64], 'dtype': 'bfloat16', 'stride': [524288, 64, 1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 16, 512, 64], 'dtype': 'bfloat16', 'stride': [524288, 64, 1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [16, 512, 1024]}, {'kind': 'shape', 'name': None, 'dims': [8192, 1024]}, {'kind': 'shape', 'name': None, 'dims': [16, 512, 1024]}, {'kind': 'shape', 'name': None, 'dims': [8192, 1024]}, {'kind': 'shape', 'name': None, 'dims': [16, 512, 1024]}, {'kind': 'shape', 'name': None, 'dims': [8192, 1024]}]}
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
