"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_d2dd7b308111
Shape hash: 9be57dd5
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6 + tmp7
        out0[tile_y, tile_x] = tmp8.to(torch.bfloat16)

def call_helion_op0_op1(arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg0_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg1_1, arg2_1, arg0_1, buf1)
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2_op3(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6 + tmp7
        out0[tile_y, tile_x] = tmp8.to(torch.bfloat16)

def call_helion_op2_op3(arg4_1: torch.Tensor, arg5_1: torch.Tensor, arg3_1: torch.Tensor, buf3: torch.Tensor) -> None:
    helion_op2_op3(arg4_1, arg5_1, arg3_1, buf3)
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
    assert_size_stride_grouped((arg1_1, arg2_1, arg0_1), ((8192, 768), (768, 768), (768,)), ((768, 1), (1, 768), (1,)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((8192, 768), (768, 1), torch.bfloat16)
        buf1 = buf0
        del buf0
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg1_1, arg2_1, arg0_1, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
        assert_size_stride_grouped((arg4_1, arg5_1, arg3_1), ((8192, 768), (768, 768), (768,)), ((768, 1), (1, 768), (1,)), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf2 = empty_strided_cuda((8192, 768), (768, 1), torch.bfloat16)
        buf3 = buf2
        del buf2
        raw_stream0 = get_raw_stream(0)
        call_helion_op2_op3(arg4_1, arg5_1, arg3_1, buf3)
        del arg3_1
        del arg4_1
        del arg5_1
    return (reinterpret_tensor(buf1, (4, 12, 2048, 64), (1572864, 64, 768, 1), 0), reinterpret_tensor(buf3, (4, 12, 2048, 64), (1572864, 64, 768, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [768, 768], 'dtype': 'bfloat16', 'stride': [1, 768], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 768], 'dtype': 'bfloat16', 'stride': [1, 768], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'shape', 'name': None, 'dims': [4, 2048, 12, 64]}, {'kind': 'shape', 'name': None, 'dims': [4, 2048, 12, 64]}], 'alias_group_nbytes': [12582912]}
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
