"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_00ec9028aed0
Shape hash: 0511b145
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1_op2(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([64, 1024, 1024]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = (tile_x.index[None, :] + -1 * tile_y.index[:, None]).to(torch.int64)
        tmp6 = hl.full([], 0, dtype=torch.int64)
        tmp7 = tmp5 > tmp6
        tmp8 = tmp7.to(torch.int64)
        tmp9 = hl.full([], 16, dtype=torch.int64)
        tmp10 = tmp8 * tmp9
        tmp11 = abs(tile_x.index[None, :] + -1 * tile_y.index[:, None]).to(torch.int64)
        tmp12 = hl.full([], 8, dtype=torch.int64)
        tmp13 = tmp11 < tmp12
        tmp14 = abs(tile_x.index[None, :] + -1 * tile_y.index[:, None]).to(torch.float32)
        tmp15 = hl.full([], 0.125, dtype=torch.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = torch.log(tmp16)
        tmp18 = hl.full([], 0.36067376022224085, dtype=torch.float32)
        tmp19 = tmp17 * tmp18
        tmp20 = hl.full([], 8.0, dtype=torch.float32)
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(torch.int64)
        tmp23 = tmp22 + tmp12
        tmp24 = hl.full([], 15, dtype=torch.int64)
        tmp25 = torch.minimum(tmp23, tmp24)
        tmp26 = torch.where(tmp13, tmp11, tmp25)
        tmp27 = tmp10 + tmp26
        tmp28 = in0[tmp27.reshape([1, tile_y, tile_x]), tile_z.index.reshape([tile_z, 1, 1]) % 8].to(torch.float32)
        tmp29 = hl.full([], 0.0, dtype=torch.float32)
        tmp30 = tmp28 + tmp29
        tmp31 = tmp4 + tmp30
        tmp32 = tmp31.to(torch.float32)
        tmp33 = tmp32.to(torch.float32)
        out0[tile_z.index.reshape([tile_z, 1, 1]) // 8, tile_z.index.reshape([tile_z, 1, 1]) % 8, tile_y.index.reshape([1, tile_y, 1]), tile_x.index.reshape([1, 1, tile_x])] = tmp33.to(torch.float32)

def call_helion_op0_op1_op2(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op0_op1_op2(arg0_1, arg1_1, arg2_1, buf2)
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
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1), ((64, 1024, 64), (64, 64, 1024), (32, 8)), ((65536, 64, 1), (65536, 1024, 1), (8, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((8, 8, 1024, 1024), (8388608, 1048576, 1024, 1), torch.float32)
        buf2 = buf1
        del buf1
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1_op2(arg0_1, arg1_1, arg2_1, buf2)
        del arg0_1
        del arg1_1
        del arg2_1
    return (buf2,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [64, 1024, 64], 'dtype': 'bfloat16', 'stride': [65536, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [64, 64, 1024], 'dtype': 'bfloat16', 'stride': [65536, 1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 8], 'dtype': 'float32', 'stride': [8, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8, 1, 1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [8, 8, 1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1024, 1024]}]}
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
