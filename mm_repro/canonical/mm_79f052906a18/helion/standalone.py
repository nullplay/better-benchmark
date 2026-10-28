"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_79f052906a18
Shape hash: ab68aafe
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op6(A: torch.Tensor, B: torch.Tensor, B1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = B1[tile_r, tile_x].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp1, tmp5)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)
        out1[tile_y, tile_x] = acc1.to(torch.bfloat16)

def call_helion_op1_op6(arg1_1: torch.Tensor, buf0: torch.Tensor, buf5: torch.Tensor, buf1: torch.Tensor, buf6: torch.Tensor) -> None:
    helion_op1_op6(arg1_1, reinterpret_tensor(buf0, (4096, 4096), (1, 4096), 0), reinterpret_tensor(buf5, (4096, 4096), (1, 4096), 0), buf1, buf6)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), 'tb', 'ctb72z7waz54thqiq5dxb55syxvjwzp5dn2wdcdknawl5q2ncpit.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 134217728}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, None)
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

@triton_heuristics.pointwise(size_hints={'x': 131072}, filename=os.path.join(cache_dir(), '77', 'c77aovydpoiidwjuvzrw53urr2rennvumxj2gibipjdv6djpspwn.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_repeat_slice_split_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 1376256}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_repeat_slice_split_unsqueeze_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 64
    x3 = xindex // 64
    x2 = xindex // 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 256 * x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (32 + 64 * x2 + x0 // 2), None, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr1 + (64 * x2 + x0 // 2), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = (x4 % 2).to(tl.int32)
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tmp4 >= tmp5
    tmp7 = (x4 % 2).to(tl.int64)
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tl.full([1], 1, tl.int64)
    tmp10 = tmp8 < tmp9
    tmp11 = tl.load(in_ptr0 + (1 + 2 * (x0 // 2) + 256 * x3), tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = -tmp11
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp10, tmp12, tmp13)
    tmp15 = tmp4 >= tmp9
    tmp16 = tl.full([1], 2, tl.int64)
    tmp17 = tmp4 < tmp16
    tmp18 = tl.load(in_ptr0 + (2 * (x0 // 2) + 256 * x3), tmp15, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.where(tmp10, tmp14, tmp18)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp19 * tmp21
    tmp23 = tmp3 + tmp22
    tl.store(out_ptr0 + (x0 + 256 * x3), tmp23, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 524288}, filename=os.path.join(cache_dir(), 'fy', 'cfyioofyezcrvihnbnimblb32px3y5xtjacmgbzqum2yoa5t2tuc.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_repeat_slice_split_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 2359296}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_repeat_slice_split_unsqueeze_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 192
    x1 = xindex // 192
    tmp0 = tl.load(in_ptr0 + (64 + x0 + 256 * x1), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + 256 * x1), tmp0, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'y': 4096, 'x': 128}, tile_hint=TileHint.DEFAULT, filename=os.path.join(cache_dir(), 'oi', 'coio4klwggjps6h6tovucl437lnjqxy6dd3ncproiwnbvd474xwq.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr2': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_permute_repeat_slice_split_unsqueeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 5505024, 'x': 0}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_permute_repeat_slice_split_unsqueeze_view_4(in_ptr0, in_ptr1, out_ptr2, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex % 256
    x2 = xindex
    y1 = yindex // 256
    y3 = yindex
    tmp0 = y0.to(tl.int32)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = y0.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1, 1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (256 * y1 + 4096 * x2 + y0), tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (32 + 64 * x2 + y0 // 2 % 32), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tl.broadcast_to(y0 % 2, [YBLOCK, XBLOCK]).to(tl.int32)
    tmp12 = tl.full([1, 1], 0, tl.int64)
    tmp13 = tmp11 >= tmp12
    tmp14 = tl.broadcast_to(y0 % 2, [YBLOCK, XBLOCK]).to(tl.int64)
    tmp15 = tmp14.to(tl.int64)
    tmp16 = tl.full([1, 1], 1, tl.int64)
    tmp17 = tmp15 < tmp16
    tmp18 = tmp17 & tmp6
    tmp19 = tl.load(in_ptr0 + (1 + 2 * (y0 // 2 % 32) + 256 * y1 + 4096 * x2), tmp18 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = -tmp19
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp18, tmp20, tmp21)
    tmp23 = tmp11 >= tmp16
    tmp24 = tl.full([1, 1], 2, tl.int64)
    tmp25 = tmp11 < tmp24
    tmp26 = tmp23 & tmp6
    tmp27 = tl.load(in_ptr0 + (2 * (y0 // 2 % 32) + 256 * y1 + 4096 * x2), tmp26 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tl.where(tmp17, tmp22, tmp27)
    tmp29 = tl.load(in_ptr1 + (64 * x2 + y0 // 2 % 32), tmp6 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp28 * tmp30
    tmp32 = tmp10 + tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp6, tmp32, tmp33)
    tmp35 = tmp0 >= tmp5
    tmp36 = tl.full([1, 1], 256, tl.int64)
    tmp37 = tmp0 < tmp36
    tmp38 = tl.load(in_ptr0 + (64 + 256 * y1 + 4096 * x2 + (-64 + y0)), tmp35 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tl.where(tmp6, tmp34, tmp38)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp40.to(tl.float32)
    tl.store(out_ptr2 + (y3 + 4096 * x2), tmp41, xmask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, 4096), (4096, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg0_1, buf0, 16777216, stream=raw_stream0)
        del arg0_1
        assert_size_stride(arg2_1, (4096, 4096), (4096, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf5 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg2_1, buf5, 16777216, stream=raw_stream0)
        del arg2_1
        assert_size_stride(arg1_1, (128, 4096), (4096, 1), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((128, 4096), (4096, 1), torch.bfloat16)
        buf6 = empty_strided_cuda((128, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op6(arg1_1, buf0, buf5, buf1, buf6)
        del arg1_1
        del buf0
        del buf5
        assert_size_stride(arg3_1, (2048, 64), (64, 1), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf4 = empty_strided_cuda((1, 128, 16, 256), (524288, 4096, 256, 1), torch.bfloat16)
        buf2 = reinterpret_tensor(buf4, (1, 128, 16, 64), (524288, 4096, 256, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_repeat_slice_split_unsqueeze_view_2.run(buf1, arg3_1, buf2, 131072, stream=raw_stream0)
        buf3 = reinterpret_tensor(buf4, (1, 128, 16, 192), (524288, 4096, 256, 1), 64)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_repeat_slice_split_unsqueeze_view_3.run(buf1, buf3, 393216, stream=raw_stream0)
        buf9 = reinterpret_tensor(buf1, (16, 256, 128), (256, 1, 4096), 0)
        del buf1
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_permute_repeat_slice_split_unsqueeze_view_4.run(buf6, arg3_1, buf9, 4096, 128, stream=raw_stream0)
        del arg3_1
        del buf6
    return (reinterpret_tensor(buf4, (16, 128, 256), (256, 4096, 1), 0), buf9, reinterpret_tensor(buf4, (16, 256, 128), (256, 1, 4096), 0), reinterpret_tensor(buf9, (16, 128, 256), (256, 4096, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'float32', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'float32', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2048, 64], 'dtype': 'float32', 'stride': [64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 4096]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 4096]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 128, 256]}, {'kind': 'shape', 'name': None, 'dims': [16, 128, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 256, 128]}, {'kind': 'shape', 'name': None, 'dims': [16, 256, 128]}]}
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
