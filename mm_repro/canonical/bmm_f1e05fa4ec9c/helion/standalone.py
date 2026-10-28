"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_f1e05fa4ec9c
Shape hash: 6e4e4947
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([4096, 49, 49]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(32):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(arg0_1, arg1_1, buf1)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op7(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([4096, 32, 49]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(49):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op7(arg7_1: torch.Tensor, buf6: torch.Tensor, buf7: torch.Tensor) -> None:
    helion_op7(arg7_1, reinterpret_tensor(buf6, (4096, 49, 49), (2401, 49, 1), 0), buf7)
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
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(size_hints={'x': 262144, 'r0_': 64}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'x5', 'cx5pk4b2375alkkjg5y3k6ddxuirkonnmsb67tfkmgzo242asp2h.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_clone_convert_element_type_div_exp_fma_index_mul_neg_permute_sub_sum_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 3211264, 'r0_': 157371144}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_clone_convert_element_type_div_exp_fma_index_mul_neg_permute_sub_sum_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 200704
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
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
    r0_3 = r0_index
    x5 = xindex
    x0 = xindex % 49
    x1 = xindex // 49 % 32
    x4 = xindex // 49
    tmp0 = tl.load(in_ptr0 + (r0_3 + 49 * x5), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_3 + 49 * x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + tl.squeeze(x5, 1), None, eviction_policy='evict_last')[:, None]
    tmp13 = tl.load(in_ptr4 + tl.squeeze(x5, 1), None, eviction_policy='evict_last')[:, None]
    tmp15 = tl.load(in_ptr5 + (r0_3 + 49 * x5), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 169, tl.int32).to(tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 169) | ~r0_mask, 'index out of bounds: 0 <= tmp6 < 169')
    tmp8 = tl.load(in_ptr2 + (x1 + 32 * tmp6), r0_mask, eviction_policy='evict_last')
    tmp9 = tmp1 + tmp8
    tmp11 = tmp9 - tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp14 = tmp12 / tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp14
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = -tmp14
    tmp23 = tl.fma(tmp22, tmp21, tmp17)
    tmp24 = tmp23.to(tl.float32)
    tl.store(out_ptr0 + (r0_3 + 49 * x0 + 2432 * x4), tmp14, r0_mask)
    tl.store(out_ptr2 + (r0_3 + 49 * x5), tmp24, r0_mask)
    tl.store(out_ptr1 + x5, tmp21, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 8192}, filename=os.path.join(cache_dir(), 'k3', 'ck3romxmrlmyoffx7hamtziyprfkccmhyvx6bgegwagvhjdk5k3l.py'), triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_full_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 43264}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_full_2(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5408
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

@triton_heuristics.reduction(size_hints={'x': 131072, 'r0_': 128}, reduction_hint=ReductionHint.DEFAULT, filename=os.path.join(cache_dir(), 'yn', 'cynnk2grbohe3s6t7iwvcc4zdnzoolyhviltqcajbbrcx7nqxqpl.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*i64', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_red_fused_convert_element_type_fma_full_index_put_mul_neg_permute_squeeze_sum_view_3', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 4, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 59026184, 'r0_': 0}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_red_fused_convert_element_type_fma_full_index_put_mul_neg_permute_squeeze_sum_view_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
    xnumel = 76832
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x2 = xindex // 2401
    x4 = xindex % 2401
    x5 = xindex // 49
    x6 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = tl.load(in_ptr0 + (x4 + 2432 * x2 + 77824 * r0_3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr1 + (x5 + 1568 * r0_3), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (x6 + 76832 * r0_3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = -tmp0
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp0
        tmp6 = tl.fma(tmp1, tmp2, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = tl.load(in_ptr3 + tl.squeeze(x4, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None]
    tmp11 = tl.full([1, 1], 169, tl.int32).to(tl.int32)
    tmp12 = tmp10 + tmp11
    tmp13 = tmp10 < 0
    tmp14 = tl.where(tmp13, tmp12, tmp10)
    tl.device_assert((0 <= tmp14) & (tmp14 < 169) | ~xmask, 'index out of bounds: 0 <= tmp14 < 169')
    tl.atomic_add(out_ptr1 + (x2 + 32 * tmp14), tmp8, xmask, sem='relaxed')
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
    assert_size_stride_grouped((arg0_1, arg1_1), ((4096, 49, 32), (4096, 32, 49)), ((1568, 32, 1), (1600, 1, 32)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((4096, 49, 49), (2401, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(arg0_1, arg1_1, buf1)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1, arg4_1, arg5_1, arg6_1), ((4096, 49, 49), (49, 49), (169, 32), (128, 32, 49, 1), (128, 32, 49, 1)), ((2401, 49, 1), (49, 1), (32, 1), (1568, 49, 1, 1), (1568, 49, 1, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        buf0 = empty_strided_cuda((128, 32, 49, 49), (77824, 2432, 49, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 32, 49, 1), (1568, 49, 1, 200704), torch.float32)
        buf6 = empty_strided_cuda((128, 32, 49, 49), (76832, 2401, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_convert_element_type_div_exp_fma_index_mul_neg_permute_sub_sum_unsqueeze_view_1.run(arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, buf1, buf0, buf2, buf6, 200704, 49, stream=raw_stream0)
        del arg2_1
        del arg4_1
        del arg5_1
        del arg6_1
        buf4 = empty_strided_cuda((169, 32), (32, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_full_2.run(buf4, 5408, stream=raw_stream0)
        raw_stream0 = get_raw_stream(0)
        triton_red_fused_convert_element_type_fma_full_index_put_mul_neg_permute_squeeze_sum_view_3.run(buf0, buf2, buf1, arg3_1, buf4, 76832, 128, stream=raw_stream0)
        del arg3_1
        del buf0
        del buf1
        del buf2
        assert_size_stride(arg7_1, (4096, 32, 49), (1600, 1, 32), 'input')
        arg7_1 = copy_if_misaligned(arg7_1)
        buf7 = empty_strided_cuda((4096, 32, 49), (1568, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op7(arg7_1, buf6, buf7)
        del arg7_1
    return (buf4, reinterpret_tensor(buf6, (4096, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf7, (128, 32, 49, 32), (50176, 1568, 1, 49), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 49, 32], 'dtype': 'bfloat16', 'stride': [1568, 32, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 32, 49], 'dtype': 'bfloat16', 'stride': [1600, 1, 32], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 49, 49], 'dtype': 'bfloat16', 'stride': [2401, 49, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [49, 49], 'dtype': 'int64', 'stride': [49, 1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 169}, 'generator': {'kind': 'index', 'low': 0, 'high': 169}}, {'kind': 'tensor', 'name': None, 'shape': [169, 32], 'dtype': 'float32', 'stride': [32, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 32, 49, 1], 'dtype': 'float32', 'stride': [1568, 49, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 32, 49, 1], 'dtype': 'float32', 'stride': [1568, 49, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 32, 49], 'dtype': 'bfloat16', 'stride': [1600, 1, 32], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [169, 32]}, {'kind': 'shape', 'name': None, 'dims': [128, 32, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [128, 32, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [49, 49, -1]}, {'kind': 'shape', 'name': None, 'dims': [2401, 32]}, {'kind': 'shape', 'name': None, 'dims': [4096, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [128, 32, 32, 49]}]}
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
