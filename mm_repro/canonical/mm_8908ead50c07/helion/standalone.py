"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_8908ead50c07
Shape hash: 6641f670
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 3072]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = B[tile_r, tile_x].to(torch.float32)
            acc = torch.addmm(acc, tmp0, tmp1)
        tmp2 = in0[tile_y, tile_x].to(torch.float32)
        tmp3 = hl.full([], 0.5, dtype=torch.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = acc * tmp4
        tmp6 = tmp2 * tmp2
        tmp7 = tmp6 * tmp2
        tmp8 = hl.full([], 0.044715, dtype=torch.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = tmp2 + tmp9
        tmp11 = hl.full([], 0.7978845608028654, dtype=torch.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = torch.tanh(tmp12)
        tmp14 = tmp13 * tmp13
        tmp15 = hl.full([], 1.0, dtype=torch.float32)
        tmp16 = tmp15 - tmp14
        tmp17 = tmp5 * tmp16
        tmp18 = tmp17 * tmp11
        tmp19 = tmp18 * tmp8
        tmp20 = hl.full([], 3.0, dtype=torch.float32)
        tmp21 = tmp6 * tmp20
        tmp22 = tmp19 * tmp21
        tmp23 = tmp18 + tmp22
        tmp24 = tmp13 + tmp15
        tmp25 = acc * tmp24
        tmp26 = tmp25 * tmp3
        tmp27 = tmp23 + tmp26
        out0[tile_y.index.reshape([tile_y, 1]) // 512, tile_y.index.reshape([tile_y, 1]) % 512, tile_x.index.reshape([1, tile_x])] = tmp27.to(torch.float32)

def call_helion_op0_op1(arg1_1: torch.Tensor, arg0_1: torch.Tensor, arg2_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg1_1, arg0_1, arg2_1, buf1)
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
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg1_1, arg0_1, arg2_1), ((16384, 768), (768, 3072), (16384, 3072)), ((768, 1), (3072, 1), (3072, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf0 = empty_strided_cuda((16384, 3072), (3072, 1), torch.float32)
        buf1 = reinterpret_tensor(buf0, (32, 512, 3072), (1572864, 3072, 1), 0)
        del buf0
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg1_1, arg0_1, arg2_1, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
    return (reinterpret_tensor(buf1, (16384, 3072), (3072, 1), 0), reinterpret_tensor(buf1, (3072, 16384), (1, 3072), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [768, 3072], 'dtype': 'float32', 'stride': [3072, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16384, 768], 'dtype': 'float32', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16384, 3072], 'dtype': 'float32', 'stride': [3072, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 3072]}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 3072]}, {'kind': 'shape', 'name': None, 'dims': [16384, 3072]}]}
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
