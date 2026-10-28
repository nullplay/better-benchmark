"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_9c5fb9a4085f
Shape hash: af122fd2
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1000, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg1_1: torch.Tensor, arg0_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg1_1, reinterpret_tensor(arg0_1, (4096, 4096), (1, 4096), 0), buf0)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1000, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op2(arg4_1: torch.Tensor, arg3_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op2(arg4_1, reinterpret_tensor(arg3_1, (4096, 1024), (1, 4096), 0), buf2)
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

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), '6g', 'c6g7fdaodcaqsy7ysuv2gikftpdvlrxnrbjadhoolzmmxcim3qgi.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_clone_convert_element_type_expand_mul_neg_permute_slice_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 40960128}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_cat_clone_convert_element_type_expand_mul_neg_permute_slice_unsqueeze_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x2 = xindex // 4096
    x0 = xindex % 128
    x4 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x3, None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + x3 % 64, None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = x2.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp6 = tl_math.cos(tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp0 * tmp7
    tmp9 = x0.to(tl.int32)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = x0.to(tl.int64)
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tl.full([1], 64, tl.int64)
    tmp15 = tmp13 < tmp14
    tmp16 = tl.load(in_ptr0 + (64 + 128 * x4 + x0), tmp15, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = -tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp15, tmp17, tmp18)
    tmp20 = tmp9 >= tmp14
    tmp21 = tl.full([1], 128, tl.int64)
    tmp22 = tmp9 < tmp21
    tmp23 = tl.load(in_ptr0 + (128 * x4 + (-64 + x0)), tmp20, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp15, tmp19, tmp23)
    tmp25 = tl_math.sin(tmp5)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp28 = tmp8 + tmp27
    tl.store(out_ptr0 + x3, tmp28, None)
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

@triton_heuristics.pointwise(size_hints={'x': 1048576}, filename=os.path.join(cache_dir(), 'jd', 'cjdgmb2cziwrdrdv4ndeznaofvcmano6xmtrv6uv7jycst3va24s.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_clone_convert_element_type_expand_mul_neg_permute_slice_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 10240128}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_cat_clone_convert_element_type_expand_mul_neg_permute_slice_unsqueeze_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x2 = xindex // 1024
    x0 = xindex % 128
    x4 = xindex // 128
    tmp0 = tl.load(in_ptr0 + x3, None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + x3 % 64, None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = x2.to(tl.float32)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp2 * tmp4
    tmp6 = tl_math.cos(tmp5)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp0 * tmp7
    tmp9 = x0.to(tl.int32)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = x0.to(tl.int64)
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tl.full([1], 64, tl.int64)
    tmp15 = tmp13 < tmp14
    tmp16 = tl.load(in_ptr0 + (64 + 128 * x4 + x0), tmp15, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = -tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp15, tmp17, tmp18)
    tmp20 = tmp9 >= tmp14
    tmp21 = tl.full([1], 128, tl.int64)
    tmp22 = tmp9 < tmp21
    tmp23 = tl.load(in_ptr0 + (128 * x4 + (-64 + x0)), tmp20, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp24 = tl.where(tmp15, tmp19, tmp23)
    tmp25 = tl_math.sin(tmp5)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp24 * tmp26
    tmp28 = tmp8 + tmp27
    tl.store(out_ptr0 + x3, tmp28, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), '6f', 'c6f3yal6a7rap7in4y2hetaqvxxghpvisorrf3eakukqtkkyyi6i.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_expand_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 18432000}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_expand_unsqueeze_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 128
    x1 = xindex // 128 % 1000
    x3 = xindex // 512000
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x3 + 1024 * x1), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + x4, tmp0, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride_grouped((arg1_1, arg0_1), ((1000, 4096), (4096, 4096)), ((4096, 1), (4096, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((1000, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg1_1, arg0_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride(arg2_1, (64,), (1,), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((1, 32, 1000, 128), (4096000, 128, 4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_convert_element_type_expand_mul_neg_permute_slice_unsqueeze_view_1.run(buf0, arg2_1, buf1, 4096000, stream=raw_stream0)
        del buf0
        assert_size_stride_grouped((arg4_1, arg3_1), ((1000, 4096), (1024, 4096)), ((4096, 1), (4096, 1)), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf2 = empty_strided_cuda((1000, 1024), (1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2(arg4_1, arg3_1, buf2)
        del arg3_1
        del arg4_1
        buf3 = empty_strided_cuda((1, 8, 1000, 128), (1024000, 128, 1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_convert_element_type_expand_mul_neg_permute_slice_unsqueeze_view_3.run(buf2, arg2_1, buf3, 1024000, stream=raw_stream0)
        del arg2_1
        del buf2
        buf4 = empty_strided_cuda((1, 8, 4, 1000, 128), (4096000, 512000, 128000, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_expand_unsqueeze_4.run(buf3, buf4, 4096000, stream=raw_stream0)
    return (buf1, buf3, reinterpret_tensor(buf4, (1, 32, 1000, 128), (4096000, 128000, 128, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [64], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, 4096]}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, -1, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, 64, 1]}, {'kind': 'shape', 'name': None, 'dims': [1, 1, 1000]}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, 2, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, -1, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, 8, 4, 1000, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, 32, 1000, 128]}], 'alias_group_nbytes': [8192000]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4]

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
