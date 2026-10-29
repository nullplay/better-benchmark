"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_mm_37e18284cce5
Shape hash: e3c33c52
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op2_op4(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = acc + tmp4
        out0[tile_y, tile_x] = tmp5.to(torch.bfloat16)
        tmp6 = in1[tile_x].to(torch.float32)
        tmp7 = acc + tmp6
        out1[tile_y, tile_x] = tmp7.to(torch.bfloat16)

def call_helion_op0_op2_op4(arg1_1: torch.Tensor, arg0_1: torch.Tensor, arg2_1: torch.Tensor, arg5_1: torch.Tensor, buf2: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op0_op2_op4(arg1_1, reinterpret_tensor(arg0_1, (1024, 1024), (1024, 1), 0), reinterpret_tensor(arg2_1, (1024,), (1,), 0), reinterpret_tensor(arg5_1, (1024,), (1,), 0), reinterpret_tensor(buf2, (8192, 1024), (1024, 1), 0), reinterpret_tensor(buf4, (8192, 1024), (1024, 1), 0))
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(arg4_1: torch.Tensor, arg3_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(arg4_1, reinterpret_tensor(arg3_1, (1024, 1024), (1024, 1), 0), buf1)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op3(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([256, 512, 512]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op3(buf2: torch.Tensor, buf1: torch.Tensor, buf3: torch.Tensor) -> None:
    helion_op3(reinterpret_tensor(buf2, (256, 512, 64), (64, 16384, 1), 0), reinterpret_tensor(buf1, (256, 64, 512), (64, 1, 16384), 0), buf3)
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
    assert_size_stride_grouped((arg1_1, arg0_1, arg2_1, arg5_1), ((8192, 1024), (1024, 16, 64), (16, 64), (16, 64)), ((1024, 1), (1024, 64, 1), (64, 1), (64, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        buf2 = empty_strided_cuda((512, 16, 16, 64), (16384, 1024, 64, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((512, 16, 16, 64), (16384, 1024, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op2_op4(arg1_1, arg0_1, arg2_1, arg5_1, buf2, buf4)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg5_1
        assert_size_stride_grouped((arg4_1, arg3_1), ((8192, 1024), (1024, 16, 64)), ((1024, 1), (1024, 64, 1)), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((8192, 1024), (1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(arg4_1, arg3_1, buf1)
        del arg3_1
        del arg4_1
        buf3 = empty_strided_cuda((256, 512, 512), (262144, 512, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op3(buf2, buf1, buf3)
        del buf1
        del buf2
    return (reinterpret_tensor(buf3, (16, 16, 512, 512), (4194304, 262144, 512, 1), 0), reinterpret_tensor(buf4, (256, 512, 64), (64, 16384, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1024, 16, 64], 'dtype': 'bfloat16', 'stride': [1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 64], 'dtype': 'bfloat16', 'stride': [64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024, 16, 64], 'dtype': 'bfloat16', 'stride': [1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 64], 'dtype': 'bfloat16', 'stride': [64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 1, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 16, 64, 1]}, {'kind': 'shape', 'name': None, 'dims': [256, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 1, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 16, 64, 1]}, {'kind': 'shape', 'name': None, 'dims': [256, 64, 512]}, {'kind': 'shape', 'name': None, 'dims': [16, 16, 512, 1, 512]}, {'kind': 'shape', 'name': None, 'dims': [16, 16, 512, 512]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 16, 64, 1]}, {'kind': 'shape', 'name': None, 'dims': [256, 512, 64]}], 'alias_group_nbytes': [16777216]}
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
