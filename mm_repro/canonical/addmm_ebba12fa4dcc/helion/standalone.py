"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_ebba12fa4dcc
Shape hash: 07c5fece
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op3_op4(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([5, 5]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6 + tmp7
        out0[tile_y, tile_x] = tmp8.to(torch.bfloat16)

def call_helion_op3_op4(buf2: torch.Tensor, arg5_1: torch.Tensor, arg4_1: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op3_op4(reinterpret_tensor(buf2, (5, 64), (64, 1), 0), arg5_1, arg4_1, buf4)
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

@triton_heuristics.pointwise(size_hints={'x': 64}, filename=os.path.join(cache_dir(), '67', 'c67nzilfo7fakzwzsfojpvg4s3ticofoysk66tna4szmrgwgb5hl.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_relu_var_mean_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 1664}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_relu_var_mean_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (64 + x0), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (128 + x0), xmask).to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (192 + x0), xmask).to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (256 + x0), xmask).to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0, tl.PropagateNan.ALL)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tl.maximum(tmp1, tmp4, tl.PropagateNan.ALL)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp3 + tmp6
    tmp9 = tl.maximum(tmp1, tmp8, tl.PropagateNan.ALL)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp7 + tmp10
    tmp13 = tl.maximum(tmp1, tmp12, tl.PropagateNan.ALL)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp11 + tmp14
    tmp17 = tl.maximum(tmp1, tmp16, tl.PropagateNan.ALL)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp15 + tmp18
    tmp20 = tl.full([1], 5.0, tl.float32)
    tmp21 = tmp19 / tmp20
    tmp22 = tmp3 - tmp21
    tmp23 = tmp22 * tmp22
    tmp24 = tmp6 - tmp21
    tmp25 = tmp24 * tmp24
    tmp26 = tmp23 + tmp25
    tmp27 = tmp10 - tmp21
    tmp28 = tmp27 * tmp27
    tmp29 = tmp26 + tmp28
    tmp30 = tmp14 - tmp21
    tmp31 = tmp30 * tmp30
    tmp32 = tmp29 + tmp31
    tmp33 = tmp18 - tmp21
    tmp34 = tmp33 * tmp33
    tmp35 = tmp32 + tmp34
    tmp36 = tmp35 / tmp20
    tl.store(out_ptr0 + x0, tmp21, xmask)
    tl.store(out_ptr1 + x0, tmp36, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 512}, filename=os.path.join(cache_dir(), 'p2', 'cp2kr5wjqm3kedyilqbycjgdvysmugjmyjmosrj5mef2lxi2tjfc.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_convert_element_type_mul_relu_rsqrt_sub_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 2688}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_convert_element_type_mul_relu_rsqrt_sub_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + x2, xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr1 + x0, xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + x0, xmask, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr4 + x0, xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0, tl.PropagateNan.ALL)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = tl.full([1], 1e-05, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp5 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp17, xmask)
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

@triton_heuristics.pointwise(size_hints={'x': 32}, filename=os.path.join(cache_dir(), '2a', 'c2aoblp4kkmnolckx6bqbtyrnv37e2ousq7ufexkjt3ljjuo455b.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_log_prepare_softmax_online_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 250}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_log_prepare_softmax_online_sub_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 25
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 5
    tmp0 = tl.load(in_ptr0 + x2, xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + 5 * x1, xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (1 + 5 * x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr0 + (2 + 5 * x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp10 = tl.load(in_ptr0 + (3 + 5 * x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr0 + (4 + 5 * x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.maximum(tmp3, tmp5, tl.PropagateNan.ALL)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.maximum(tmp6, tmp8, tl.PropagateNan.ALL)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.maximum(tmp9, tmp11, tl.PropagateNan.ALL)
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.maximum(tmp12, tmp14, tl.PropagateNan.ALL)
    tmp16 = tmp1 - tmp15
    tmp17 = tmp3 - tmp15
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tmp5 - tmp15
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tmp18 + tmp20
    tmp22 = tmp8 - tmp15
    tmp23 = libdevice.exp(tmp22)
    tmp24 = tmp21 + tmp23
    tmp25 = tmp11 - tmp15
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tmp24 + tmp26
    tmp28 = tmp14 - tmp15
    tmp29 = libdevice.exp(tmp28)
    tmp30 = tmp27 + tmp29
    tmp31 = tl_math.log(tmp30)
    tmp32 = tmp16 - tmp31
    tl.store(out_ptr0 + x2, tmp32, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 1}, filename=os.path.join(cache_dir(), '5b', 'c5bmvb3hca3d2zci2bbs2yt42aovldi7q337nvxdw6wcwsofbulv.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_div_full_gather_ne_neg_sum_view_where_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_div_full_gather_ne_neg_sum_view_where_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp12 = tl.load(in_ptr0 + 1)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK])
    tmp22 = tl.load(in_ptr0 + 2)
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK])
    tmp32 = tl.load(in_ptr0 + 3)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp42 = tl.load(in_ptr0 + 4)
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK])
    tmp2 = tl.full([1], -100, tl.int64)
    tmp3 = tmp1 != tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tl.device_assert((0 <= tmp5) & (tmp5 < 5), 'index out of bounds: 0 <= tmp5 < 5')
    tmp7 = tl.load(in_ptr1 + tmp5, None, eviction_policy='evict_last')
    tmp8 = tmp7.to(tl.float32)
    tmp9 = -tmp8
    tmp10 = tl.full([1], 0.0, tl.float32)
    tmp11 = tl.where(tmp3, tmp9, tmp10)
    tmp14 = tmp13 != tmp2
    tmp15 = tl.where(tmp14, tmp13, tmp4)
    tl.device_assert((0 <= tmp15) & (tmp15 < 5), 'index out of bounds: 0 <= tmp15 < 5')
    tmp17 = tl.load(in_ptr1 + (5 + tmp15), None, eviction_policy='evict_last')
    tmp18 = tmp17.to(tl.float32)
    tmp19 = -tmp18
    tmp20 = tl.where(tmp14, tmp19, tmp10)
    tmp21 = tmp11 + tmp20
    tmp24 = tmp23 != tmp2
    tmp25 = tl.where(tmp24, tmp23, tmp4)
    tl.device_assert((0 <= tmp25) & (tmp25 < 5), 'index out of bounds: 0 <= tmp25 < 5')
    tmp27 = tl.load(in_ptr1 + (10 + tmp25), None, eviction_policy='evict_last')
    tmp28 = tmp27.to(tl.float32)
    tmp29 = -tmp28
    tmp30 = tl.where(tmp24, tmp29, tmp10)
    tmp31 = tmp21 + tmp30
    tmp34 = tmp33 != tmp2
    tmp35 = tl.where(tmp34, tmp33, tmp4)
    tl.device_assert((0 <= tmp35) & (tmp35 < 5), 'index out of bounds: 0 <= tmp35 < 5')
    tmp37 = tl.load(in_ptr1 + (15 + tmp35), None, eviction_policy='evict_last')
    tmp38 = tmp37.to(tl.float32)
    tmp39 = -tmp38
    tmp40 = tl.where(tmp34, tmp39, tmp10)
    tmp41 = tmp31 + tmp40
    tmp44 = tmp43 != tmp2
    tmp45 = tl.where(tmp44, tmp43, tmp4)
    tl.device_assert((0 <= tmp45) & (tmp45 < 5), 'index out of bounds: 0 <= tmp45 < 5')
    tmp47 = tl.load(in_ptr1 + (20 + tmp45), None, eviction_policy='evict_last')
    tmp48 = tmp47.to(tl.float32)
    tmp49 = -tmp48
    tmp50 = tl.where(tmp44, tmp49, tmp10)
    tmp51 = tmp41 + tmp50
    tmp52 = tmp3.to(tl.int64)
    tmp53 = tmp14.to(tl.int64)
    tmp54 = tmp52 + tmp53
    tmp55 = tmp24.to(tl.int64)
    tmp56 = tmp54 + tmp55
    tmp57 = tmp34.to(tl.int64)
    tmp58 = tmp56 + tmp57
    tmp59 = tmp44.to(tl.int64)
    tmp60 = tmp58 + tmp59
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp51 / tmp61
    tl.store(in_out_ptr0 + tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK), tmp62, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride(arg0_1, (5, 64, 1, 1), (64, 1, 1, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        buf1 = empty_strided_cuda((1, 64, 1, 1), (64, 1, 64, 64), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_relu_var_mean_0.run(arg0_1, buf0, buf1, 64, stream=raw_stream0)
        assert_size_stride_grouped((arg2_1, arg3_1), ((64,), (64,)), ((1,), (1,)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf2 = empty_strided_cuda((5, 64, 1, 1), (64, 1, 320, 320), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_convert_element_type_mul_relu_rsqrt_sub_view_1.run(arg0_1, buf0, buf1, arg2_1, arg3_1, buf2, 320, stream=raw_stream0)
        del arg0_1
        del arg2_1
        del arg3_1
        del buf0
        del buf1
        assert_size_stride_grouped((arg5_1, arg4_1), ((64, 5), (5,)), ((1, 64), (1,)), 'input')
        arg5_1 = copy_if_misaligned(arg5_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf3 = empty_strided_cuda((5, 5), (5, 1), torch.bfloat16)
        buf4 = buf3
        del buf3
        raw_stream0 = get_raw_stream(0)
        call_helion_op3_op4(buf2, arg5_1, arg4_1, buf4)
        del arg4_1
        del arg5_1
        del buf2
        buf5 = empty_strided_cuda((5, 5), (5, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_log_prepare_softmax_online_sub_3.run(buf4, buf5, 25, stream=raw_stream0)
        assert_size_stride(arg1_1, (5,), (1,), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf6 = empty_strided_cuda((), (), torch.bfloat16)
        buf7 = buf6
        del buf6
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_div_full_gather_ne_neg_sum_view_where_4.run(buf7, arg1_1, buf5, 1, stream=raw_stream0)
        del arg1_1
        del buf5
    return (buf4, buf7)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [5, 64, 1, 1], 'dtype': 'bfloat16', 'stride': [64, 1, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [5], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 5}, 'generator': {'kind': 'index', 'low': 0, 'high': 5}}, {'kind': 'tensor', 'name': None, 'shape': [64], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [64], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [64, 5], 'dtype': 'bfloat16', 'stride': [1, 64], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [5], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [64, 1, 1]}, {'kind': 'shape', 'name': None, 'dims': [64, 1, 1]}, {'kind': 'shape', 'name': None, 'dims': [5, -1]}, {'kind': 'shape', 'name': None, 'dims': [5, 1]}, {'kind': 'shape', 'name': None, 'dims': [5]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 5, 4]

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
