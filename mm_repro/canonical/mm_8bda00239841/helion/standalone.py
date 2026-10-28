"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_8bda00239841
Shape hash: f8483fbd
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1_op2_op3_op4_op5(A: torch.Tensor, B: torch.Tensor, A1: torch.Tensor, B1: torch.Tensor, A2: torch.Tensor, B2: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, in3: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 512]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc2 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(384):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = A1[tile_y, tile_r].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            tmp6 = B1[tile_r, tile_x].to(torch.float32)
            tmp7 = tmp6.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp5, tmp7)
            tmp8 = A2[tile_y, tile_r].to(torch.float32)
            tmp9 = tmp8.to(torch.bfloat16)
            tmp10 = B2[tile_r, tile_x].to(torch.float32)
            tmp11 = tmp10.to(torch.bfloat16)
            acc2 = torch.addmm(acc2, tmp9, tmp11)
        tmp12 = acc.to(torch.float32)
        tmp13 = acc1.to(torch.float32)
        tmp14 = tmp12 + tmp13
        tmp15 = acc2.to(torch.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = in0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.bool)
        tmp18 = tmp17.to(torch.float32)
        tmp19 = in1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.float32)
        tmp20 = tmp18 * tmp19
        tmp21 = hl.full([], 1.1111111111111112, dtype=torch.float32)
        tmp22 = tmp20 * tmp21
        tmp23 = in2[tile_y.index // 128, tile_y.index % 128, 0].to(torch.float32)
        tmp24 = tmp22 * tmp23.reshape([tile_y, 1])
        tmp25 = tmp16 * tmp24
        out0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp25.to(torch.float32)
        tmp26 = acc.to(torch.float32)
        tmp27 = acc1.to(torch.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = acc2.to(torch.float32)
        tmp30 = tmp28 + tmp29
        tmp31 = in3[tile_x].to(torch.float32)
        tmp32 = tmp30 * tmp31.reshape([1, tile_x])
        tmp33 = in0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.bool)
        tmp34 = tmp33.to(torch.float32)
        tmp35 = in1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.float32)
        tmp36 = tmp34 * tmp35
        tmp37 = hl.full([], 1.1111111111111112, dtype=torch.float32)
        tmp38 = tmp36 * tmp37
        tmp39 = tmp32 * tmp38
        out1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp39.to(torch.float32)
        tmp40 = acc.to(torch.float32)
        tmp41 = acc1.to(torch.float32)
        tmp42 = tmp40 + tmp41
        tmp43 = acc2.to(torch.float32)
        tmp44 = tmp42 + tmp43
        tmp45 = in3[tile_x].to(torch.float32)
        tmp46 = tmp44 * tmp45.reshape([1, tile_x])
        tmp47 = in2[tile_y.index // 128, tile_y.index % 128, 0].to(torch.float32)
        tmp48 = tmp46 * tmp47.reshape([tile_y, 1])
        out2[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp48.to(torch.float32)

def call_helion_op0_op1_op2_op3_op4_op5(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg3_1: torch.Tensor, arg4_1: torch.Tensor, arg5_1: torch.Tensor, arg7_1: torch.Tensor, arg8_1: torch.Tensor, arg9_1: torch.Tensor, arg6_1: torch.Tensor, buf3: torch.Tensor, buf4: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op0_op1_op2_op3_op4_op5(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg7_1, arg8_1, arg9_1, arg6_1, buf3, buf4, buf5)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg7_1, arg8_1, arg9_1, arg6_1), ((4096, 384), (384, 512), (4096, 384), (384, 512), (4096, 384), (384, 512), (32, 128, 512), (32, 128, 512), (32, 128, 1), (512,)), ((384, 1), (512, 1), (384, 1), (512, 1), (384, 1), (512, 1), (65536, 512, 1), (65536, 512, 1), (128, 1, 1), (1,)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg7_1 = copy_if_misaligned(arg7_1)
        arg8_1 = copy_if_misaligned(arg8_1)
        arg9_1 = copy_if_misaligned(arg9_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        buf3 = empty_strided_cuda((32, 128, 512), (65536, 512, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 128, 512), (65536, 512, 1), torch.float32)
        buf5 = empty_strided_cuda((32, 128, 512), (65536, 512, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1_op2_op3_op4_op5(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg7_1, arg8_1, arg9_1, arg6_1, buf3, buf4, buf5)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del arg8_1
        del arg9_1
    return (buf3, buf4, buf5)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 384], 'dtype': 'bfloat16', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 384], 'dtype': 'bfloat16', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 384], 'dtype': 'bfloat16', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 512], 'dtype': 'bool', 'stride': [65536, 512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 512], 'dtype': 'float32', 'stride': [65536, 512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 1], 'dtype': 'float32', 'stride': [128, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 512]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 512]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 512]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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
