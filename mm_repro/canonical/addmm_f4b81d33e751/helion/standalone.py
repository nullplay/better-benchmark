"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_f4b81d33e751
Shape hash: 96af11e1
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op8(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([401408, 384]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op8(buf7: torch.Tensor, arg5_1: torch.Tensor, buf8: torch.Tensor) -> None:
    helion_op8(reinterpret_tensor(buf7, (401408, 128), (128, 1), 0), reinterpret_tensor(arg5_1, (128, 384), (1, 128), 0), buf8)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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

@triton_heuristics.persistent_reduction(size_hints={'x': 524288, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'gd', 'cgdqbhek4udepbvtmnatexlzugqkiw65ri7puneik5r4de3du4ex.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr5': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_clone_convert_element_type_mul_permute_rsqrt_sub_var_mean_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 8, 'autotune_hints': (), 'tiling_scores': {'x': 0, 'r0_': 308282368}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_clone_convert_element_type_mul_permute_rsqrt_sub_var_mean_view_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 401408
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x2 = xindex % 7
    x3 = xindex // 7 % 8
    x4 = xindex // 56 % 7
    x5 = xindex // 392
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128 * x0), None, eviction_policy='evict_first').to(tl.float32)
    tmp21 = tl.load(in_ptr1 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :].to(tl.float32)
    tmp24 = tl.load(in_ptr2 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :].to(tl.float32)
    tmp45 = tl.load(in_ptr3 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :].to(tl.float32)
    tmp48 = tl.load(in_ptr4 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :].to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp6 = tl.sum(tmp4, 1)[:, None].to(tl.float32)
    tmp7 = tl.full([1, 1], 128, tl.int32).to(tl.float32)
    tmp8 = tmp6 / tmp7
    tmp9 = tmp2 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp14 = tmp1 - tmp8
    tmp15 = tl.full([1, 1], 128.0, tl.float32)
    tmp16 = tmp13 / tmp15
    tmp17 = tl.full([1, 1], 1e-05, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = libdevice.rsqrt(tmp18)
    tmp20 = tmp14 * tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp20 * tmp22
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp23 + tmp25
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, R0_BLOCK])
    tmp31 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp33 = tl.sum(tmp31, 1)[:, None].to(tl.float32)
    tmp34 = tmp33 / tmp7
    tmp35 = tmp29 - tmp34
    tmp36 = tmp35 * tmp35
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp40 = tmp28 - tmp34
    tmp41 = tmp39 / tmp15
    tmp42 = tmp41 + tmp17
    tmp43 = libdevice.rsqrt(tmp42)
    tmp44 = tmp40 * tmp43
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 * tmp46
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp47 + tmp49
    tmp51 = tmp50.to(tl.float32)
    tl.store(out_ptr5 + (r0_1 + 128 * x2 + 896 * x4 + 6272 * x3 + 50176 * x5), tmp51, None)
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

@triton_heuristics.pointwise(size_hints={'x': 67108864}, filename=os.path.join(cache_dir(), 'um', 'cumr65oj4zf4lgwg2ukepcibf6mjv3nidnksj7o2ajihqfxtjff6.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 308282112}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 32
    x2 = xindex // 1568
    x1 = xindex // 32 % 49
    x3 = xindex % 1568
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * (x2 % 4)), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0 + 32 * (x2 % 4) + 384 * x1 + 18816 * (x2 // 4)), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tmp6 = tl.full([1], 0.1767766952966369, tl.float32)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x3 + 1792 * x2), tmp7, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'zq', 'czq6xlq3ermo7mwj56jlqxiqpfer2udarux5m65rdqbnc2mcevtv.py'), triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 29360128}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_3(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 7340032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 224
    x1 = xindex // 224
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0 + 1792 * x1), tmp0, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'y': 1048576, 'x': 64}, tile_hint=TileHint.DEFAULT, filename=os.path.join(cache_dir(), '2t', 'c2thub5psn2fq66bv34bgwiknxglwb3sgx4f7mlyzigt2zyx7icw.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 102760448, 'x': 205521664}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_4(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 1048576
    xnumel = 49
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex % 32
    y1 = yindex // 32
    x2 = xindex
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + tl.squeeze(128 + y0 + 32 * (y1 % 4), 1), tl.squeeze(ymask, 1), eviction_policy='evict_last')[:, None].to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (128 + y0 + 32 * (y1 % 4) + 384 * x2 + 18816 * (y1 // 4)), xmask & ymask).to(tl.float32)
    tmp1 = tl.full([1, 1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tl.store(out_ptr0 + (x2 + 56 * y3), tmp5, xmask & ymask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'ex', 'cexq6puyocfduxwrss347tp6ntwljewpabainen6eb7573sevezj.py'), triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 29360128}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_5(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 7340032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 7
    x1 = xindex // 7
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0 + 56 * x1), tmp0, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 67108864}, filename=os.path.join(cache_dir(), 'ta', 'ctaxmq2rzetnrh5qvtluwx54523jtfhx7p6fwwzuryepi67kar7v.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 308282112}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_6(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 32
    x2 = xindex // 1568
    x1 = xindex // 32 % 49
    x3 = xindex % 1568
    tmp0 = tl.load(in_ptr0 + (256 + x0 + 32 * (x2 % 4)), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (256 + x0 + 32 * (x2 % 4) + 384 * x1 + 18816 * (x2 // 4)), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tl.store(out_ptr0 + (x3 + 1792 * x2), tmp5, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1, arg2_1, arg3_1, arg4_1), ((128, 128, 56, 56), (128,), (128,), (128,), (128,)), ((401408, 1, 7168, 128), (1,), (1,), (1,), (1,)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf7 = empty_strided_cuda((128, 8, 8, 7, 7, 128), (401408, 50176, 6272, 896, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_convert_element_type_mul_permute_rsqrt_sub_var_mean_view_0.run(arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, buf7, 401408, 128, stream=raw_stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        assert_size_stride(arg5_1, (384, 128), (128, 1), 'input')
        arg5_1 = copy_if_misaligned(arg5_1)
        buf8 = empty_strided_cuda((401408, 384), (384, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op8(buf7, arg5_1, buf8)
        del arg5_1
        del buf7
        assert_size_stride(arg6_1, (384,), (1,), 'input')
        arg6_1 = copy_if_misaligned(arg6_1)
        buf11 = empty_strided_cuda((32768, 56, 32), (1792, 32, 1), torch.bfloat16)
        buf9 = reinterpret_tensor(buf11, (32768, 49, 32), (1792, 32, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_2.run(arg6_1, buf8, buf9, 51380224, stream=raw_stream0)
        buf10 = reinterpret_tensor(buf11, (32768, 7, 32), (1792, 32, 1), 1568)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_3.run(buf10, 7340032, stream=raw_stream0)
        buf14 = empty_strided_cuda((32768, 32, 56), (1792, 56, 1), torch.bfloat16)
        buf12 = reinterpret_tensor(buf14, (32768, 32, 49), (1792, 56, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_4.run(arg6_1, buf8, buf12, 1048576, 49, stream=raw_stream0)
        buf13 = reinterpret_tensor(buf14, (32768, 32, 7), (1792, 56, 1), 49)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_5.run(buf13, 7340032, stream=raw_stream0)
        buf17 = empty_strided_cuda((32768, 56, 32), (1792, 32, 1), torch.bfloat16)
        buf15 = reinterpret_tensor(buf17, (32768, 49, 32), (1792, 32, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_6.run(arg6_1, buf8, buf15, 51380224, stream=raw_stream0)
        del arg6_1
        del buf8
        buf16 = reinterpret_tensor(buf17, (32768, 7, 32), (1792, 32, 1), 1568)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_constant_pad_nd_convert_element_type_mul_permute_rsqrt_sub_unbind_var_mean_view_3.run(buf16, 7340032, stream=raw_stream0)
    return (buf11, buf14, buf17)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [128, 128, 56, 56], 'dtype': 'bfloat16', 'stride': [401408, 1, 7168, 128], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [128, 8, 7, 8, 7, 128]}, {'kind': 'shape', 'name': None, 'dims': [-1, 7, 7, 128]}, {'kind': 'shape', 'name': None, 'dims': [-1, 49, 128]}, {'kind': 'shape', 'name': None, 'dims': [401408, 128]}, {'kind': 'shape', 'name': None, 'dims': [8192, 49, 384]}, {'kind': 'shape', 'name': None, 'dims': [8192, 49, 3, 4, -1]}, {'kind': 'shape', 'name': None, 'dims': [8192, 4, 49, 32]}, {'kind': 'shape', 'name': None, 'dims': [32768, 49, 32]}, {'kind': 'shape', 'name': None, 'dims': [0, 0, 0, 7, 0, 0]}, {'kind': 'shape', 'name': None, 'dims': [8192, 4, 32, 49]}, {'kind': 'shape', 'name': None, 'dims': [32768, 32, 49]}, {'kind': 'shape', 'name': None, 'dims': [0, 7, 0, 0, 0, 0]}, {'kind': 'shape', 'name': None, 'dims': [8192, 4, 49, 32]}, {'kind': 'shape', 'name': None, 'dims': [32768, 49, 32]}, {'kind': 'shape', 'name': None, 'dims': [0, 0, 0, 7, 0, 0]}]}
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
