"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_77709072e15a
Shape hash: 34d75d82
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        out0[tile_y, tile_x] = tmp8.permute(1, 0).to(torch.bfloat16)

def call_helion_op0_op1(arg2_1: torch.Tensor, arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg2_1, reinterpret_tensor(arg0_1, (4096, 1024), (1, 4096), 0), arg1_1, buf1)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2_op3(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        out0[tile_y, tile_x] = tmp8.permute(1, 0).to(torch.bfloat16)

def call_helion_op2_op3(arg5_1: torch.Tensor, arg3_1: torch.Tensor, arg4_1: torch.Tensor, buf3: torch.Tensor) -> None:
    helion_op2_op3(arg5_1, reinterpret_tensor(arg3_1, (1024, 1024), (1, 1024), 0), arg4_1, buf3)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op4_op5(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        out0[tile_y, tile_x] = tmp8.permute(1, 0).to(torch.bfloat16)

def call_helion_op4_op5(arg8_1: torch.Tensor, arg6_1: torch.Tensor, arg7_1: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op4_op5(arg8_1, reinterpret_tensor(arg6_1, (1024, 1024), (1, 1024), 0), arg7_1, buf5)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op6_op7(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        out0[tile_y, tile_x] = tmp8.permute(1, 0).to(torch.bfloat16)

def call_helion_op6_op7(arg11_1: torch.Tensor, arg9_1: torch.Tensor, arg10_1: torch.Tensor, buf7: torch.Tensor) -> None:
    helion_op6_op7(arg11_1, reinterpret_tensor(arg9_1, (1024, 1024), (1, 1024), 0), arg10_1, buf7)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1 = args
    args.clear()
    assert_size_stride_grouped((arg2_1, arg0_1, arg1_1), ((8192, 4096), (1024, 4096), (1024,)), ((4096, 1), (4096, 1), (1,)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((8192, 1024), (1024, 1), torch.bfloat16)
        buf1 = buf0
        del buf0
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg2_1, arg0_1, arg1_1, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
        assert_size_stride_grouped((arg5_1, arg3_1, arg4_1), ((8192, 1024), (1024, 1024), (1024,)), ((1024, 1), (1024, 1), (1,)), 'input')
        arg5_1 = copy_if_misaligned(arg5_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf2 = empty_strided_cuda((8192, 1024), (1024, 1), torch.bfloat16)
        buf3 = buf2
        del buf2
        raw_stream0 = get_raw_stream(0)
        call_helion_op2_op3(arg5_1, arg3_1, arg4_1, buf3)
        del arg3_1
        del arg4_1
        del arg5_1
        assert_size_stride_grouped((arg8_1, arg6_1, arg7_1), ((8192, 1024), (1024, 1024), (1024,)), ((1024, 1), (1024, 1), (1,)), 'input')
        arg8_1 = copy_if_misaligned(arg8_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        arg7_1 = copy_if_misaligned(arg7_1)
        buf4 = empty_strided_cuda((8192, 1024), (1024, 1), torch.bfloat16)
        buf5 = buf4
        del buf4
        raw_stream0 = get_raw_stream(0)
        call_helion_op4_op5(arg8_1, arg6_1, arg7_1, buf5)
        del arg6_1
        del arg7_1
        del arg8_1
        assert_size_stride_grouped((arg11_1, arg9_1, arg10_1), ((8192, 1024), (1024, 1024), (1024,)), ((1024, 1), (1024, 1), (1,)), 'input')
        arg11_1 = copy_if_misaligned(arg11_1)
        arg9_1 = copy_if_misaligned(arg9_1)
        arg10_1 = copy_if_misaligned(arg10_1)
        buf6 = empty_strided_cuda((8192, 1024), (1024, 1), torch.bfloat16)
        buf7 = buf6
        del buf6
        raw_stream0 = get_raw_stream(0)
        call_helion_op6_op7(arg11_1, arg9_1, arg10_1, buf7)
        del arg10_1
        del arg11_1
        del arg9_1
    return (reinterpret_tensor(buf1, (64, 128, 1024), (131072, 1024, 1), 0), reinterpret_tensor(buf3, (64, 16, 128, 64), (131072, 64, 1024, 1), 0), reinterpret_tensor(buf5, (64, 16, 128, 64), (131072, 64, 1024, 1), 0), reinterpret_tensor(buf7, (64, 16, 128, 64), (131072, 64, 1024, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1024, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'shape', 'name': None, 'dims': [64, 128, 1024]}, {'kind': 'shape', 'name': None, 'dims': [64, 128, 1024]}, {'kind': 'shape', 'name': None, 'dims': [64, 128, -1, 64]}, {'kind': 'shape', 'name': None, 'dims': [64, 128, 1024]}, {'kind': 'shape', 'name': None, 'dims': [64, 128, -1, 64]}, {'kind': 'shape', 'name': None, 'dims': [64, 128, 1024]}, {'kind': 'shape', 'name': None, 'dims': [64, 128, -1, 64]}], 'alias_group_nbytes': [16777216]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

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
