"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_b6f951c37c5b
Shape hash: e10ec79b
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op5_op6(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([2, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp4.to(torch.float32)

def call_helion_op5_op6(buf4: torch.Tensor, arg13_1: torch.Tensor, buf6: torch.Tensor) -> None:
    helion_op5_op6(reinterpret_tensor(buf4, (2, 128), (1, 2), 0), arg13_1, buf6)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 1, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'oe', 'coek22h7of2msio7vjdma2v4zxzq34wzn5b5puikbjeyvir3uwgp.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*i1', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_clamp_max_clamp_min_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'r0_': 1024}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_clamp_max_clamp_min_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1
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
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [1, 1])
    tmp14 = tl.load(in_ptr1 + 0)
    tmp15 = tl.broadcast_to(tmp14, [1, 1])
    tmp18 = tl.load(in_ptr2 + 0).to(tl.int1)
    tmp19 = tl.broadcast_to(tmp18, [1, 1])
    tmp30 = tl.load(in_ptr3 + tl.squeeze(r0_0, 0), None, eviction_policy='evict_first')[None, :].to(tl.float32)
    tmp31 = tl.load(in_ptr4 + tl.squeeze(r0_0, 0), None, eviction_policy='evict_first')[None, :].to(tl.float32)
    tmp33 = tl.load(in_ptr5 + 0)
    tmp34 = tl.broadcast_to(tmp33, [1, 1])
    tmp36 = tl.load(in_ptr6 + 0)
    tmp37 = tl.broadcast_to(tmp36, [1, 1])
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tl.maximum(tmp1, tmp2, tl.PropagateNan.ALL)
    tmp4 = tl.full([1, 1], 128, tl.int64)
    tmp5 = tl.minimum(tmp3, tmp4, tl.PropagateNan.ALL)
    tmp6 = tmp5 != tmp4
    tmp7 = tl.where(tmp6, tmp5, tmp2)
    tmp8 = r0_0.to(tl.int64)
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tmp7 == tmp9
    tmp11 = tl.full([1, 1], -1.0, tl.float32)
    tmp12 = tl.full([1, 1], 0.0, tl.float32)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp16 = tl.full([1, 1], 0.5, tl.float32)
    tmp17 = tmp15 * tmp16
    tmp20 = tmp19.to(tl.int64)
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp17 / tmp21
    tmp23 = tl.where(tmp6, tmp22, tmp12)
    tmp24 = tmp13 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp32 = tmp31.to(tl.float32)
    tmp35 = tmp32 - tmp34
    tmp38 = tmp35 - tmp37
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp39.to(tl.float32)
    tmp41 = libdevice.exp(tmp40)
    tmp42 = tmp41 * tmp29
    tmp43 = tmp26 - tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp30 + tmp44
    tl.store(out_ptr1 + tl.broadcast_to(r0_0, [XBLOCK, R0_BLOCK]), tmp45, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 256}, filename=os.path.join(cache_dir(), 'hm', 'chm3u6unzn3tlso4wjauosom7wubdmg4yswtrhfeyscliclp63ye.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_cat_unsqueeze_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 1536}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_cat_unsqueeze_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = xindex // 2
    x2 = xindex
    tmp0 = x0.to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1], 1, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + x1, tmp6 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1], 2, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + x1, tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tl.where(tmp6, tmp7, tmp11)
    tl.store(out_ptr0 + x2, tmp12, xmask)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1 = args
    args.clear()
    assert_size_stride_grouped((arg8_1, arg0_1, arg7_1, arg12_1, arg9_1, arg10_1, arg11_1), ((1,), (), (1,), (1, 128), (1, 128), (1, 1), (1, 1)), ((1,), (), (1,), (128, 1), (128, 1), (1, 1), (1, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg8_1 = copy_if_misaligned(arg8_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg7_1 = copy_if_misaligned(arg7_1)
        arg12_1 = copy_if_misaligned(arg12_1)
        arg9_1 = copy_if_misaligned(arg9_1)
        arg10_1 = copy_if_misaligned(arg10_1)
        arg11_1 = copy_if_misaligned(arg11_1)
        buf1 = empty_strided_cuda((1, 128), (128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_max_clamp_min_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0.run(arg8_1, arg0_1, arg7_1, arg12_1, arg9_1, arg10_1, arg11_1, buf1, 1, 128, stream=raw_stream0)
        del arg10_1
        del arg11_1
        del arg12_1
        del arg7_1
        del arg8_1
        del arg9_1
        assert_size_stride_grouped((arg2_1, arg1_1, arg6_1, arg3_1, arg4_1, arg5_1), ((1,), (1,), (1, 128), (1, 128), (1, 1), (1, 1)), ((1,), (1,), (128, 1), (128, 1), (1, 1), (1, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        buf3 = empty_strided_cuda((1, 128), (128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_max_clamp_min_convert_element_type_div_eq_exp_expand_full_iota_mul_ne_scalar_tensor_sub_sum_unsqueeze_view_where_0.run(arg2_1, arg0_1, arg1_1, arg6_1, arg3_1, arg4_1, arg5_1, buf3, 1, 128, stream=raw_stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf4 = empty_strided_cuda((1, 128, 2), (256, 2, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_cat_unsqueeze_1.run(buf1, buf3, buf4, 256, stream=raw_stream0)
        del buf1
        del buf3
        assert_size_stride(arg13_1, (128, 4096), (4096, 1), 'input')
        arg13_1 = copy_if_misaligned(arg13_1)
        buf6 = empty_strided_cuda((2, 4096), (4096, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op5_op6(buf4, arg13_1, buf6)
        del arg13_1
        del buf4
    return (buf6,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [], 'dtype': 'float32', 'stride': [], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1], 'dtype': 'bool', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 1], 'dtype': 'float32', 'stride': [1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 1], 'dtype': 'float32', 'stride': [1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1], 'dtype': 'bool', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 1], 'dtype': 'float32', 'stride': [1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 1], 'dtype': 'float32', 'stride': [1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, 128]}, {'kind': 'shape', 'name': None, 'dims': [128, 2]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

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
