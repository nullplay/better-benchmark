"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_fe0ba35016eb
Shape hash: cbc46054
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([384, 256, 768]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg0_1, arg1_1, buf0)
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

@triton_heuristics.pointwise(size_hints={'x': 67108864}, filename=os.path.join(cache_dir(), 'cd', 'ccdqyqcrssowrqfuzbthjkgt6lwarw4xzq3yuii4wut2oniihd3z.py'), triton_meta={'signature': {'in_ptr0': '*i1', 'in_ptr1': '*bf16', 'in_ptr2': '*i1', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_convert_element_type_div_exp_full_mul_permute_squeeze_sub_unsqueeze_view_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 656384000}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_convert_element_type_div_exp_full_mul_permute_squeeze_sub_unsqueeze_view_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 50429952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x5 = xindex // 6156
    x0 = xindex % 513
    x2 = xindex // 6156 % 1024
    x1 = xindex // 513 % 12
    x3 = xindex // 6303744
    x4 = xindex % 6156
    x6 = xindex // 513
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + x5, None, eviction_policy='evict_last').to(tl.int1)
    tmp23 = tl.load(in_ptr2 + (x4 + 6272 * x5), None).to(tl.int1)
    tmp31 = tl.load(in_ptr3 + (x0 + 513 * x2 + 525312 * x1 + 6303744 * x3), None).to(tl.float32)
    tmp33 = tl.load(in_ptr4 + x6, None, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr5 + x6, None, eviction_policy='evict_last')
    tmp1 = x0.to(tl.int32)
    tmp2 = tl.full([1], 770, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = (x0 + 770 * (x2 % 256)).to(tl.int64)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tl.full([1], 196864, tl.int64)
    tmp7 = tmp5 < tmp6
    tmp8 = tmp7 & tmp3
    tmp9 = ((x0 + 770 * (x2 % 256)) % 769).to(tl.int64)
    tmp10 = tmp9.to(tl.int64)
    tmp11 = tl.full([1], 768, tl.int64)
    tmp12 = tmp10 < tmp11
    tmp13 = tmp12 & tmp8
    tmp14 = tl.load(in_ptr1 + (768 * ((x0 + 770 * (x2 % 256)) // 769) + 196608 * (x2 // 256) + 786432 * x1 + 9437184 * x3 + (x0 + 770 * (x2 % 256)) % 769), tmp13, other=0.0).to(tl.float32)
    tmp15 = tl.full([1], 0.0, tl.float32)
    tmp16 = tl.where(tmp12, tmp14, tmp15)
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = tl.full([1], 0.0, tl.float32)
    tmp20 = tl.where(tmp7, tmp18, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp3, tmp20, tmp21)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.full([1], 1.1111111111111112, tl.float32)
    tmp26 = tmp24 * tmp25
    tmp27 = tmp22 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.full([1], 0.0, tl.float32)
    tmp30 = tl.where(tmp0, tmp29, tmp28)
    tmp32 = tmp31.to(tl.float32)
    tmp34 = tmp32 - tmp33
    tmp35 = libdevice.exp(tmp34)
    tmp37 = tmp35 / tmp36
    tmp38 = tmp30 * tmp37
    tl.store(out_ptr0 + x7, tmp38, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((384, 256, 64), (384, 64, 768)), ((16384, 64, 1), (49152, 1, 64)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((384, 256, 768), (196608, 768, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg3_1, arg2_1, arg4_1, arg5_1, arg6_1), ((8, 1024), (8, 1024, 12, 513), (8, 1024, 12, 513), (8, 1024, 12, 1), (8, 1024, 12, 1)), ((1024, 1), (6422528, 6272, 513, 1), (6303744, 513, 525312, 1), (12288, 12, 1, 1), (12288, 12, 1, 1)), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        buf1 = empty_strided_cuda((8, 1024, 12, 513), (6303744, 6156, 513, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_constant_pad_nd_convert_element_type_div_exp_full_mul_permute_squeeze_sub_unsqueeze_view_where_1.run(arg3_1, buf0, arg2_1, arg4_1, arg5_1, arg6_1, buf1, 50429952, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del buf0
    return (buf1,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [384, 256, 64], 'dtype': 'bfloat16', 'stride': [16384, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 64, 768], 'dtype': 'bfloat16', 'stride': [49152, 1, 64], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 1024, 12, 513], 'dtype': 'bool', 'stride': [6422528, 6272, 513, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 1024], 'dtype': 'bool', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 1024, 12, 513], 'dtype': 'bfloat16', 'stride': [6303744, 513, 525312, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 1024, 12, 1], 'dtype': 'float32', 'stride': [12288, 12, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 1024, 12, 1], 'dtype': 'float32', 'stride': [12288, 12, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [96, 4, 256, 768, 1]}, {'kind': 'shape', 'name': None, 'dims': [96, 4, 256, 769]}, {'kind': 'shape', 'name': None, 'dims': [96, 4, 196864]}, {'kind': 'shape', 'name': None, 'dims': [96, 4, 197120]}, {'kind': 'shape', 'name': None, 'dims': [96, 4, 256, 770]}, {'kind': 'shape', 'name': None, 'dims': [0, -257]}, {'kind': 'shape', 'name': None, 'dims': [8, 12, 1024, 513]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6]

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
