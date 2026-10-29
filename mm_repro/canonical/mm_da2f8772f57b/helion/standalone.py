"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_da2f8772f57b
Shape hash: 2dbe9ecf
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 2]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg1_1, arg2_1, buf0)
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

@triton_heuristics.pointwise(size_hints={'x': 16}, filename=os.path.join(cache_dir(), '3p', 'c3pui7jle3m2arqjbselmdcjkpse2ppasxbj5hynjvrkytt4uszf.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_index_iota_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 128}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_index_iota_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2
    x0 = xindex % 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 1024, tl.int32).to(tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 1024) | ~xmask, 'index out of bounds: 0 <= tmp4 < 1024')
    tmp6 = tl.load(in_ptr1 + (x0 + 2 * tmp4 + 2048 * x1), xmask).to(tl.float32)
    tl.store(out_ptr0 + x2, tmp6, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(size_hints={'x': 1, 'r0_': 8}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'ge', 'cge7xsrngvneffykgxt6inm3dht5mcymenk7n6n7o5qtde7djbfy.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'xnumel': 'constexpr', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_convert_element_type_div_full_gather_log_ne_neg_prepare_softmax_online_sub_sum_view_where_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 2, 'autotune_hints': (), 'tiling_scores': {'r0_': 64}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_convert_element_type_div_full_gather_log_ne_neg_prepare_softmax_online_sub_sum_view_where_2(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1
    r0_numel = 8
    R0_BLOCK: tl.constexpr = 8
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_0 = r0_index
    tmp0 = tl.load(in_ptr0 + tl.squeeze(r0_0, 0), tl.squeeze(r0_mask, 0), eviction_policy='evict_first', other=0.0)[None, :]
    tmp8 = tl.load(in_ptr1 + tl.squeeze(2 * r0_0, 0), tl.squeeze(r0_mask, 0), eviction_policy='evict_last', other=0.0)[None, :].to(tl.float32)
    tmp10 = tl.load(in_ptr1 + tl.squeeze(1 + 2 * r0_0, 0), tl.squeeze(r0_mask, 0), eviction_policy='evict_last', other=0.0)[None, :].to(tl.float32)
    tmp1 = tl.full([1, 1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1, 1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.device_assert((0 <= tmp4) & (tmp4 < 2) | ~r0_mask, 'index out of bounds: 0 <= tmp4 < 2')
    tmp6 = tl.load(in_ptr1 + tl.squeeze(tmp4 + 2 * r0_0, 0), tl.squeeze(r0_mask, 0), eviction_policy='evict_last', other=0.0)[None, :].to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.maximum(tmp9, tmp11, tl.PropagateNan.ALL)
    tmp13 = tmp7 - tmp12
    tmp14 = tmp9 - tmp12
    tmp15 = libdevice.exp(tmp14)
    tmp16 = tmp11 - tmp12
    tmp17 = libdevice.exp(tmp16)
    tmp18 = tmp15 + tmp17
    tmp19 = tl_math.log(tmp18)
    tmp20 = tmp13 - tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = -tmp21
    tmp23 = tl.full([1, 1], 0.0, tl.float32)
    tmp24 = tl.where(tmp2, tmp22, tmp23)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = tmp2.to(tl.int64)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.where(r0_mask, tmp30, 0)
    tmp33 = tl.sum(tmp32, 1)[:, None].to(tl.int64)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp28 / tmp34
    tl.store(in_out_ptr0 + tl.full([1, 1], 0, tl.int32).broadcast_to(XBLOCK, 1), tmp35, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg1_1, arg2_1), ((8192, 768), (768, 2)), ((768, 1), (1, 768)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf0 = empty_strided_cuda((8192, 2), (2, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg1_1, arg2_1, buf0)
        del arg1_1
        del arg2_1
        assert_size_stride(arg3_1, (8,), (1,), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((8, 2), (2, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_index_iota_view_1.run(arg3_1, buf0, buf1, 16, stream=raw_stream0)
        del arg3_1
        del buf0
        assert_size_stride(arg0_1, (8,), (1,), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf2 = empty_strided_cuda((), (), torch.bfloat16)
        buf4 = buf2
        del buf2
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_convert_element_type_div_full_gather_log_ne_neg_prepare_softmax_online_sub_sum_view_where_2.run(buf4, arg0_1, buf1, 1, 8, stream=raw_stream0)
        del arg0_1
    return (buf1, buf4)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [8], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 2}, 'generator': {'kind': 'index', 'low': 0, 'high': 2}}, {'kind': 'tensor', 'name': None, 'shape': [768, 2], 'dtype': 'bfloat16', 'stride': [1, 768], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 1024}, 'generator': {'kind': 'index', 'low': 0, 'high': 1024}}, {'kind': 'shape', 'name': None, 'dims': [8, 1024, 2]}, {'kind': 'shape', 'name': None, 'dims': [8, 1]}, {'kind': 'shape', 'name': None, 'dims': [8]}]}
COMPILED_INPUT_ORDER = [0, 2, 1, 3]

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
