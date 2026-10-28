"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_585257e60d0b
Shape hash: 10b53b64
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op4(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 8]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op4(arg2_1: torch.Tensor, buf3: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op4(arg2_1, buf3, buf4)
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

@triton_heuristics.pointwise(size_hints={'y': 4096, 'x': 2}, tile_hint=TileHint.SQUARE, filename=os.path.join(cache_dir(), 'ri', 'cri7r7q4fdey2tmmhrqrsacmlxhisfy67gvyumey5ywontphxr6k.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_constant_pad_nd_convert_element_type_permute_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 32768, 'x': 8192}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_constant_pad_nd_convert_element_type_permute_0(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 4096
    xnumel = 2
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4096 * x1), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr1 + (x1 + 8 * y0), tmp1, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 32768}, filename=os.path.join(cache_dir(), 'y2', 'cy27njfeyukiz4d2v36xis7apk5mtlsoyxu45dmu2ypv6dpady6a.py'), triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_constant_pad_nd_convert_element_type_permute_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 98304}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_constant_pad_nd_convert_element_type_permute_1(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 6
    x1 = xindex // 6
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0 + 8 * x1), tmp0, None)
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

@triton_heuristics.pointwise(size_hints={'x': 128}, filename=os.path.join(cache_dir(), 'bv', 'cbvdw3luro3ebmpngfp4w6b4su5iyj6gjzi3wi7ox3gnvljpk4u7.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_cat_clone_convert_element_type_full_slice_split_squeeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 1536}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_cat_clone_convert_element_type_full_slice_split_squeeze_view_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp19 = tl.load(in_ptr1 + 8 * x0, xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tmp0 >= tmp0
    tmp2 = tl.full([1], 2, tl.int64)
    tmp3 = tmp0 < tmp2
    tmp4 = tl.load(in_ptr0 + 0)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK])
    tmp6 = tl.where(tmp3, tmp5, 0.0)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp3, tmp7, tmp8)
    tmp10 = tmp0 >= tmp2
    tmp11 = tl.full([1], 8, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.full([1], 0.0, tl.float32)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp10, tmp13, tmp14)
    tmp16 = tl.where(tmp3, tmp9, tmp15)
    tmp17 = tl.full([1], 1.0, tl.float32)
    tmp18 = tmp17 * tmp16
    tmp20 = tmp17 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp21, xmask)
    tl.store(out_ptr1 + x0, tmp22, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 128}, filename=os.path.join(cache_dir(), 'yw', 'cyw5rm366c25woh3dq5vls4muxurxvyyeeuzzf5jzongh7tix6gp.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_cat_clone_convert_element_type_full_slice_split_squeeze_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 1536}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_cat_clone_convert_element_type_full_slice_split_squeeze_view_4(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp20 = tl.load(in_ptr1 + (1 + 8 * x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = tl.full([1], 1, tl.int64)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + 1)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK])
    tmp7 = tl.where(tmp4, tmp6, 0.0)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 8, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.full([1], 0.0, tl.float32)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp11, tmp14, tmp15)
    tmp17 = tl.where(tmp4, tmp10, tmp16)
    tmp18 = tl.full([1], 1.0, tl.float32)
    tmp19 = tmp18 * tmp17
    tmp21 = tmp18 * tmp20
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp22, xmask)
    tl.store(out_ptr1 + x0, tmp23, xmask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg1_1, (2, 4096), (4096, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf3 = empty_strided_cuda((4096, 8), (8, 1), torch.bfloat16)
        buf1 = reinterpret_tensor(buf3, (4096, 2), (8, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_convert_element_type_permute_0.run(arg1_1, buf1, 4096, 2, stream=raw_stream0)
        del arg1_1
        buf2 = reinterpret_tensor(buf3, (4096, 6), (8, 1), 2)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_convert_element_type_permute_1.run(buf2, 24576, stream=raw_stream0)
        assert_size_stride(arg2_1, (128, 4096), (4096, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf4 = empty_strided_cuda((128, 8), (8, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op4(arg2_1, buf3, buf4)
        del arg2_1
        del buf1
        del buf2
        del buf3
        assert_size_stride(arg0_1, (2,), (1,), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf5 = empty_strided_cuda((1, 128), (128, 1), torch.bfloat16)
        buf6 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_cat_clone_convert_element_type_full_slice_split_squeeze_view_3.run(arg0_1, buf4, buf5, buf6, 128, stream=raw_stream0)
        buf7 = empty_strided_cuda((1, 128), (128, 1), torch.bfloat16)
        buf8 = empty_strided_cuda((1, 128), (128, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_cat_clone_convert_element_type_full_slice_split_squeeze_view_4.run(arg0_1, buf4, buf7, buf8, 128, stream=raw_stream0)
        del arg0_1
        del buf4
    return (buf5, buf7, buf6, buf8)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [2], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2, 4096], 'dtype': 'float32', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [0, 6, 0, 0]}, {'kind': 'shape', 'name': None, 'dims': [6]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 2]}]}
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
