"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_584ca96b2e27
Shape hash: 10fac315
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([768, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(8192):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp4.to(torch.float32)

def call_helion_op0_op1(arg1_1: torch.Tensor, arg0_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg1_1, reinterpret_tensor(arg0_1, (8192, 768), (768, 1), 0), buf1)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2_op4_op3_op5(A: torch.Tensor, B: torch.Tensor, A1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([768, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(8192):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = A1[tile_y, tile_r].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp5, tmp3)
        tmp6 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp6.to(torch.float32)
        tmp7 = acc1.to(torch.float32)
        out1[tile_y, tile_x] = tmp7.to(torch.float32)

def call_helion_op2_op4_op3_op5(arg2_1: torch.Tensor, arg4_1: torch.Tensor, arg3_1: torch.Tensor, buf3: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op2_op4_op3_op5(reinterpret_tensor(arg2_1, (768, 8192), (1, 768), 0), arg4_1, reinterpret_tensor(arg3_1, (768, 8192), (1, 768), 0), buf3, buf5)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride_grouped((arg1_1, arg0_1), ((768, 8192), (4, 12, 2048, 64)), ((1, 768), (1572864, 64, 768, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf1 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg1_1, arg0_1, buf1)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg4_1, arg3_1), ((4, 12, 2048, 64), (8192, 768), (4, 12, 2048, 64)), ((1572864, 64, 768, 1), (768, 1), (1572864, 64, 768, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf3 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        buf5 = empty_strided_cuda((768, 768), (768, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2_op4_op3_op5(arg2_1, arg4_1, arg3_1, buf3, buf5)
        del arg2_1
        del arg3_1
        del arg4_1
    return (buf1, buf3, buf5)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4, 12, 2048, 64], 'dtype': 'bfloat16', 'stride': [1572864, 64, 768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 8192], 'dtype': 'bfloat16', 'stride': [1, 768], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4, 12, 2048, 64], 'dtype': 'bfloat16', 'stride': [1572864, 64, 768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4, 12, 2048, 64], 'dtype': 'bfloat16', 'stride': [1572864, 64, 768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [4, 2048, -1]}, {'kind': 'shape', 'name': None, 'dims': [8192, 768]}, {'kind': 'shape', 'name': None, 'dims': [4, 2048, 768]}, {'kind': 'shape', 'name': None, 'dims': [4, 2048, 768]}, {'kind': 'shape', 'name': None, 'dims': [8192, 768]}, {'kind': 'shape', 'name': None, 'dims': [8192, 768]}]}
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
