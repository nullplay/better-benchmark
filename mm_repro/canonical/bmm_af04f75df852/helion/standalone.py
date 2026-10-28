"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_af04f75df852
Shape hash: 56b3f831
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([16, 256, 128]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
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

@triton_heuristics.pointwise(size_hints={'y': 1024, 'x': 128}, tile_hint=TileHint.DEFAULT, filename=os.path.join(cache_dir(), 'ut', 'cutveu5x7svpsjh4uhkxkgnpy5limmv3hy6ezvykrm4bfbeuzuxv.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_add_clone_convert_element_type_expand_full_mul_neg_permute_select_slice_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 16384, 'x': 1310720}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_clone_convert_element_type_expand_full_mul_neg_permute_select_slice_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 1024
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex % 64
    x2 = xindex
    y1 = yindex // 64
    y3 = yindex
    tmp31 = tl.load(in_ptr0 + (x2 + 128 * y0 + 32768 * y1), xmask).to(tl.float32)
    tmp34 = tl.load(in_ptr2 + (64 * x2 + y0 // 2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = y0.to(tl.int64)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1, 1], 1, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = ((-1 + y0) % 2).to(tl.int64)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tl.full([1, 1], 0, tl.int64)
    tmp7 = tmp5 == tmp6
    tmp8 = tmp3 & tmp7
    tmp9 = tl.load(in_ptr0 + (x2 + 256 * triton_helpers.div_floor_integer(-1 + y0, 2) + 32768 * y1), tmp8 & xmask, other=0.0).to(tl.float32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (64 * x2 + (-1 + y0) // 2 % 32), tmp8 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = -tmp13
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp8, tmp14, tmp15)
    tmp17 = tl.full([1, 1], 0.0, tl.float32)
    tmp18 = tl.where(tmp8, tmp16, tmp17)
    tmp19 = (y3 % 2).to(tl.int64)
    tmp20 = tmp19.to(tl.int64)
    tmp21 = tmp20 == tmp6
    tmp22 = tl.load(in_ptr0 + (128 + x2 + 256 * (y0 // 2) + 32768 * y1), tmp21 & xmask, other=0.0).to(tl.float32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (64 * x2 + y0 // 2), tmp21 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp26 = tmp24 * tmp25
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp21, tmp26, tmp27)
    tmp29 = tl.where(tmp21, tmp28, tmp17)
    tmp30 = tmp18 + tmp29
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp30 + tmp35
    tl.store(out_ptr0 + (x2 + 128 * y3), tmp36, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'y': 4096, 'x': 128}, tile_hint=TileHint.DEFAULT, filename=os.path.join(cache_dir(), 'nk', 'cnkdxucdfm4tthh6xjg2kdih3podsgxuexta4viufmq6l5twvwln.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_add_convert_element_type_full_permute_slice_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 2097152, 'x': 1310720}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_convert_element_type_full_permute_slice_view_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
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
    y3 = yindex
    y1 = yindex // 256
    tmp0 = y0.to(tl.int64)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1, 1], 64, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.load(in_ptr0 + (x2 + 128 * y3), tmp3 & xmask, other=0.0).to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp3, tmp6, tmp7)
    tmp9 = tl.full([1, 1], 0.0, tl.float32)
    tmp10 = tl.where(tmp3, tmp8, tmp9)
    tmp11 = tmp1 < tmp2
    tmp12 = tl.load(in_ptr1 + (x2 + 128 * y0 + 8192 * y1), tmp11 & xmask, other=0.0).to(tl.float32)
    tmp13 = tl.where(tmp11, tmp12, tmp9)
    tmp14 = tmp10 + tmp13
    tl.store(out_ptr0 + (y3 + 4096 * x2), tmp14, xmask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((16, 256, 128), (16, 128, 128)), ((256, 1, 4096), (16384, 128, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((16, 256, 128), (32768, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1), ((1, 128, 1, 32, 1), (1, 128, 1, 32, 1)), ((8192, 64, 32, 1, 1), (8192, 64, 32, 1, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((1, 128, 16, 64), (131072, 1, 8192, 128), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convert_element_type_expand_full_mul_neg_permute_select_slice_view_1.run(buf0, arg2_1, arg3_1, buf1, 1024, 128, stream=raw_stream0)
        del arg2_1
        del arg3_1
        buf2 = empty_strided_cuda((1, 128, 16, 256), (524288, 4096, 256, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_convert_element_type_full_permute_slice_view_2.run(buf0, buf1, buf2, 4096, 128, stream=raw_stream0)
        del buf0
        del buf1
    return (reinterpret_tensor(buf2, (128, 4096), (4096, 1), 0), reinterpret_tensor(buf2, (4096, 128), (1, 4096), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [16, 256, 128], 'dtype': 'bfloat16', 'stride': [256, 1, 4096], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 128, 128], 'dtype': 'bfloat16', 'stride': [16384, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128, 1, 32, 1], 'dtype': 'bfloat16', 'stride': [8192, 64, 32, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128, 1, 32, 1], 'dtype': 'bfloat16', 'stride': [8192, 64, 32, 1, 1], 'device': 'cuda', 'storage_offset': 32}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 256, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 4096]}, {'kind': 'shape', 'name': None, 'dims': [128, 4096]}]}
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
