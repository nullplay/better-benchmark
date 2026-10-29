"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_f8db7a170d53
Shape hash: 4f000d26
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([32768, 384]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg0_1, arg1_1, buf0)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), 'gp', 'cgpwg5dzwo763dl3ntjqcah52buazxroo3mxcsxbdwizhno64pgd.py'), triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_masked_index_put_accumulate_add_bitwise_and_constant_pad_nd_convert_element_type_full_ge_lt_ne_slice_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 31254528}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused__unsafe_masked_index_put_accumulate_add_bitwise_and_constant_pad_nd_convert_element_type_full_ge_lt_ne_slice_view_1(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3906816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + x0, tmp0, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), '66', 'c66zck7sztymwn572ufg2qhbnp6x4cljfe2wwxyzvv6nzxwr6ya7.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__unsafe_masked_index_put_accumulate_add_bitwise_and_constant_pad_nd_convert_element_type_full_ge_lt_ne_slice_view_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 25427968}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused__unsafe_masked_index_put_accumulate_add_bitwise_and_constant_pad_nd_convert_element_type_full_ge_lt_ne_slice_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex // 128
    x0 = xindex % 128
    x1 = xindex // 128 % 128
    tmp0 = tl.load(in_ptr0 + x3, None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr1 + (128 + x0 + 384 * x3), None).to(tl.float32)
    tmp1 = tl.full([1], 30521, tl.int64)
    tmp2 = tl.minimum(tmp1, tmp0, tl.PropagateNan.ALL)
    tmp3 = tl.full([1], -30522, tl.int64)
    tmp4 = tl.maximum(tmp3, tmp2, tl.PropagateNan.ALL)
    tmp5 = tl.full([XBLOCK], 30522, tl.int32).to(tl.int32)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp4 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp4)
    tmp9 = tl.full([1], 0, tl.int64)
    tmp10 = tmp0 >= tmp9
    tmp11 = tl.full([1], 30522, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp10 & tmp12
    tmp14 = tmp0 != tmp9
    tmp15 = tmp13 & tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = x1.to(tl.int64)
    tmp19 = tmp18.to(tl.int64)
    tmp20 = tl.full([1], 127, tl.int64)
    tmp21 = tmp19 < tmp20
    tmp22 = (1 + x1).to(tl.int32)
    tmp23 = tl.full([1], 0, tl.int64)
    tmp24 = tmp22 >= tmp23
    tmp25 = tmp24 & tmp21
    tmp26 = tl.load(in_ptr1 + (640 + x0 + 384 * x3), tmp25, other=0.0).to(tl.float32)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp25, tmp27, tmp28)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp21, tmp29, tmp30)
    tmp32 = tl.full([1], 0.0, tl.float32)
    tmp33 = tl.where(tmp21, tmp31, tmp32)
    tmp34 = tmp17 + tmp33
    tmp35 = tl.full([1], 1, tl.int64)
    tmp36 = tmp19 >= tmp35
    tmp37 = (-1 + x1).to(tl.int32)
    tmp38 = tl.full([1], 128, tl.int64)
    tmp39 = tmp37 < tmp38
    tmp40 = tmp39 & tmp36
    tmp41 = tl.load(in_ptr1 + (-384 + x0 + 384 * x3), tmp40, other=0.0).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp40, tmp42, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp36, tmp44, tmp45)
    tmp47 = tl.where(tmp36, tmp46, tmp32)
    tmp48 = tmp34 + tmp47
    tmp49 = tl.where(tmp15, tmp48, tmp32)
    tl.atomic_add(out_ptr0 + (x0 + 128 * tmp8), tmp49, None, sem='relaxed')
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((32768, 512), (512, 384)), ((512, 1), (384, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((32768, 384), (384, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((30522, 128), (128, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_masked_index_put_accumulate_add_bitwise_and_constant_pad_nd_convert_element_type_full_ge_lt_ne_slice_view_1.run(buf1, 3906816, stream=raw_stream0)
        assert_size_stride(arg2_1, (256, 128), (128, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__unsafe_masked_index_put_accumulate_add_bitwise_and_constant_pad_nd_convert_element_type_full_ge_lt_ne_slice_view_2.run(arg2_1, buf0, buf1, 4194304, stream=raw_stream0)
        del arg2_1
        del buf0
    return (buf1,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [32768, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 384], 'dtype': 'bfloat16', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [256, 128], 'dtype': 'int64', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [256, 128, 384]}, {'kind': 'shape', 'name': None, 'dims': [256, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [256, 128, 1]}, {'kind': 'shape', 'name': None, 'dims': [30522, 128]}]}
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
