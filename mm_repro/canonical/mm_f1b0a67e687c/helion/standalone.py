"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_f1b0a67e687c
Shape hash: b850c895
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1_op2_op3_op4(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor, out3: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([2048, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(3072):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = torch.neg(tmp4)
        tmp6 = in0[tile_x].to(torch.float32)
        tmp7 = in1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.float32)
        tmp8 = tmp6.reshape([tile_x, 1]) * tmp7.permute(1, 0)
        tmp9 = in2[tile_y.index // 128, tile_y.index % 128, 0].to(torch.float32)
        tmp10 = hl.full([], 1e-06, dtype=torch.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = tmp8 / tmp11.reshape([1, tile_y])
        tmp13 = tmp12 / tmp11.reshape([1, tile_y])
        tmp14 = tmp5 * tmp13.permute(1, 0)
        out0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp14.to(torch.float32)
        tmp15 = acc.to(torch.float32)
        tmp16 = in2[tile_y.index // 128, tile_y.index % 128, 0].to(torch.float32)
        tmp17 = hl.full([], 1e-06, dtype=torch.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = tmp15 / tmp18.reshape([tile_y, 1])
        tmp20 = in1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.float32)
        tmp21 = tmp19 * tmp20
        out1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp21.to(torch.float32)
        tmp22 = acc.to(torch.float32)
        tmp23 = in2[tile_y.index // 128, tile_y.index % 128, 0].to(torch.float32)
        tmp24 = hl.full([], 1e-06, dtype=torch.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp22 / tmp25.reshape([tile_y, 1])
        tmp27 = in0[tile_x].to(torch.float32)
        tmp28 = tmp26 * tmp27.reshape([1, tile_x])
        out2[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp28.to(torch.float32)
        tmp29 = torch.neg(tmp28)
        out3[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp29.to(torch.float32)

def call_helion_op0_op1_op2_op3_op4(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg3_1: torch.Tensor, arg4_1: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor, buf3: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op0_op1_op2_op3_op4(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf1, buf2, buf3, buf4)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1, arg3_1, arg4_1), ((2048, 3072), (3072, 768), (768,), (16, 128, 768), (16, 128, 1)), ((3072, 1), (768, 1), (1,), (98304, 768, 1), (128, 1, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf1 = empty_strided_cuda((16, 128, 768), (98304, 768, 1), torch.float32)
        buf2 = empty_strided_cuda((16, 128, 768), (98304, 768, 1), torch.float32)
        buf3 = empty_strided_cuda((16, 128, 768), (98304, 768, 1), torch.float32)
        buf4 = empty_strided_cuda((16, 128, 768), (98304, 768, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1_op2_op3_op4(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf1, buf2, buf3, buf4)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
    return (buf1, buf3, buf2, buf4)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [2048, 3072], 'dtype': 'bfloat16', 'stride': [3072, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [3072, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 128, 768], 'dtype': 'float32', 'stride': [98304, 768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 128, 1], 'dtype': 'float32', 'stride': [128, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [16, 128, 768]}]}
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
