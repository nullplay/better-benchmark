"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_2b2b51805f34
Shape hash: 57679356
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0_op1_op2_op3_op4_op5(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, in3: torch.Tensor, in4: torch.Tensor, in5: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor, out3: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 2560]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(10240):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = in0[tile_x].to(torch.float32)
        tmp6 = tmp4 * tmp5.reshape([1, tile_x])
        out0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp6.to(torch.float32)
        tmp7 = hl.full([], 2560.0, dtype=torch.float32)
        tmp8 = tmp6 * tmp7
        out1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp8.to(torch.float32)
        tmp9 = in1[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.float32)
        tmp10 = in2[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])].to(torch.bool)
        tmp11 = tmp10.to(torch.float32)
        tmp12 = in3[tile_y, tile_x].to(torch.float32)
        tmp13 = tmp11 * tmp12
        tmp14 = hl.full([], 1.1111111111111112, dtype=torch.float32)
        tmp15 = tmp13 * tmp14
        tmp16 = tmp15.to(torch.float32)
        tmp17 = tmp9 + tmp16
        tmp18 = in4[tile_y.index // 128, tile_y.index % 128, 0].to(torch.float32)
        tmp19 = tmp17 - tmp18.reshape([tile_y, 1])
        tmp20 = in5[tile_y.index // 128, tile_y.index % 128, 0].to(torch.float32)
        tmp21 = tmp19 * tmp20.reshape([tile_y, 1])
        tmp22 = tmp6 * tmp21
        out2[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp22.to(torch.float32)
        tmp23 = acc.to(torch.float32)
        tmp24 = tmp23 * tmp21
        out3[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp24.to(torch.float32)

def call_helion_op0_op1_op2_op3_op4_op5(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, arg5_1: torch.Tensor, arg4_1: torch.Tensor, arg3_1: torch.Tensor, arg6_1: torch.Tensor, arg7_1: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor, buf4: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op0_op1_op2_op3_op4_op5(arg0_1, arg1_1, arg2_1, arg5_1, arg4_1, arg3_1, arg6_1, arg7_1, buf1, buf2, buf4, buf5)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1, arg5_1, arg4_1, arg3_1, arg6_1, arg7_1), ((4096, 10240), (10240, 2560), (2560,), (32, 128, 2560), (32, 128, 2560), (4096, 2560), (32, 128, 1), (32, 128, 1)), ((10240, 1), (2560, 1), (1,), (327680, 2560, 1), (327680, 2560, 1), (2560, 1), (128, 1, 1), (128, 1, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        arg7_1 = copy_if_misaligned(arg7_1)
        buf1 = empty_strided_cuda((32, 128, 2560), (327680, 2560, 1), torch.float32)
        buf2 = empty_strided_cuda((32, 128, 2560), (327680, 2560, 1), torch.float32)
        buf4 = empty_strided_cuda((32, 128, 2560), (327680, 2560, 1), torch.float32)
        buf5 = empty_strided_cuda((32, 128, 2560), (327680, 2560, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0_op1_op2_op3_op4_op5(arg0_1, arg1_1, arg2_1, arg5_1, arg4_1, arg3_1, arg6_1, arg7_1, buf1, buf2, buf4, buf5)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
    return (buf1, buf2, buf4, buf5)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 10240], 'dtype': 'bfloat16', 'stride': [10240, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [10240, 2560], 'dtype': 'bfloat16', 'stride': [2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2560], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 2560], 'dtype': 'bfloat16', 'stride': [2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 2560], 'dtype': 'bool', 'stride': [327680, 2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 2560], 'dtype': 'float32', 'stride': [327680, 2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 1], 'dtype': 'float32', 'stride': [128, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 1], 'dtype': 'float32', 'stride': [128, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 2560]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 2560]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]

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
