"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_016b3fa4dad0
Shape hash: 8f8c8443
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([384, 512, 512]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_z.index.reshape([tile_z, 1, 1]) // 12, tile_z.index.reshape([tile_z, 1, 1]) % 12, tile_y.index.reshape([1, tile_y, 1]), tile_x.index.reshape([1, 1, tile_x])].to(torch.bool)
        tmp5 = tmp4.to(torch.float32)
        tmp6 = hl.full([], 1.1111111111111112, dtype=torch.float32)
        tmp7 = tmp5 * tmp6
        tmp8 = acc * tmp7
        tmp9 = tmp8.to(torch.float32)
        tmp10 = in1[tile_z.index.reshape([tile_z, 1, 1]) // 12, tile_z.index.reshape([tile_z, 1, 1]) % 12, tile_y.index.reshape([1, tile_y, 1]), tile_x.index.reshape([1, 1, tile_x])].to(torch.float32)
        tmp11 = tmp10.to(torch.float32)
        tmp12 = tmp9 * tmp11
        out0[tile_z.index.reshape([tile_z, 1, 1]) // 12, tile_z.index.reshape([tile_z, 1, 1]) % 12, tile_y.index.reshape([1, tile_y, 1]), tile_x.index.reshape([1, 1, tile_x])] = tmp12.to(torch.float32)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg3_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, arg2_1, arg3_1, buf1)
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
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1, arg3_1), ((384, 512, 64), (384, 64, 512), (32, 12, 512, 512), (32, 12, 512, 512)), ((32768, 64, 1), (32768, 1, 64), (3145728, 262144, 512, 1), (3145728, 262144, 512, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((32, 12, 512, 512), (3145728, 262144, 512, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1(arg0_1, arg1_1, arg2_1, arg3_1, buf1)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
    return (buf1,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [384, 512, 64], 'dtype': 'bfloat16', 'stride': [32768, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 64, 512], 'dtype': 'bfloat16', 'stride': [32768, 1, 64], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 12, 512, 512], 'dtype': 'bool', 'stride': [3145728, 262144, 512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 12, 512, 512], 'dtype': 'bfloat16', 'stride': [3145728, 262144, 512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 12, 512, 512]}]}
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
