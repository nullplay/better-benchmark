"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_mm_80a78ef7a4c5
Shape hash: b7281db3
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 384]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg2_1: torch.Tensor, arg0_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg2_1, reinterpret_tensor(arg0_1, (768, 384), (1, 768), 0), buf0)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 54]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(384):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op2(arg4_1: torch.Tensor, arg3_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op2(arg4_1, reinterpret_tensor(arg3_1, (384, 54), (1, 384), 0), buf2)
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

@triton_heuristics.pointwise(size_hints={'y': 8388608, 'x': 16}, tile_hint=TileHint.DEFAULT, filename=os.path.join(cache_dir(), 'ny', 'cnylz6ccr5yrt6cpjeauertd5z3si7cumrs5em7lj7ruqeixjjhy.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'kernel_name': 'triton_poi_fused_add_addmm_clone_constant_pad_nd_full_index_iota_permute_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 12582912, 'x': 226493184}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_clone_constant_pad_nd_full_index_iota_permute_unsqueeze_view_1(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 6291456
    xnumel = 9
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y1 = yindex // 384 % 512
    y0 = yindex % 384
    y4 = yindex
    tmp0 = (-4 + x3 + y1).to(tl.int32)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + tl.broadcast_to(y0, [YBLOCK, XBLOCK]), tmp5 & xmask & ymask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tl.full([1, 1], 1.0, tl.float32)
    tmp8 = tmp7 * tmp6
    tmp9 = tl.load(in_ptr1 + (-1536 + y4 + 384 * x3), tmp5 & xmask & ymask, other=0.0).to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tmp8 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp5, tmp11, tmp12)
    tl.store(out_ptr0 + (x3 + 9 * y4), tmp13, xmask & ymask)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 131072, 'r0_': 16}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'zk', 'czkuspkjemnw563fjnwjid4ppotk5ppdh4rmtiyx4wxqqhgued26.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_convert_element_type_div_exp_expand_prepare_softmax_online_sub_view_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 0, 'r0_': 5308524}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_convert_element_type_div_exp_expand_prepare_softmax_online_sub_view_3(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 98304
    r0_numel = 9
    R0_BLOCK: tl.constexpr = 16
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 9 * x0), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1 + 9 * x0) % 54, r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(r0_mask, tmp6, float('-inf'))
    tmp9 = triton_helpers.fmax2(tmp8, 1)[:, None].to(tl.float32)
    tmp10 = tmp4 - tmp9
    tmp11 = libdevice.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tmp3 - tmp9
    tmp17 = libdevice.exp(tmp16)
    tmp18 = tmp17 / tmp15
    tmp19 = tmp18.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 9 * x0), tmp19, r0_mask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride_grouped((arg2_1, arg0_1), ((16384, 768), (384, 768)), ((768, 1), (768, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((16384, 384), (384, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg2_1, arg0_1, buf0)
        del arg0_1
        del arg2_1
        assert_size_stride(arg1_1, (384,), (1,), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((32, 512, 384, 9), (1769472, 3456, 9, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_constant_pad_nd_full_index_iota_permute_unsqueeze_view_1.run(arg1_1, buf0, buf1, 6291456, 9, stream=raw_stream0)
        del arg1_1
        del buf0
        assert_size_stride_grouped((arg4_1, arg3_1), ((16384, 384), (54, 384)), ((384, 1), (384, 1)), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf2 = empty_strided_cuda((16384, 54), (54, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2(arg4_1, arg3_1, buf2)
        del arg3_1
        del arg4_1
        assert_size_stride(arg5_1, (54,), (1,), 'input')
        arg5_1 = copy_if_misaligned(arg5_1)
        buf5 = reinterpret_tensor(buf2, (98304, 9, 1), (9, 1, 1), 0)
        del buf2
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_convert_element_type_div_exp_expand_prepare_softmax_online_sub_view_3.run(buf5, arg5_1, 98304, 9, stream=raw_stream0)
        del arg5_1
    return (reinterpret_tensor(buf1, (98304, 64, 9), (576, 9, 1), 0), buf5)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [384, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16384, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [54, 384], 'dtype': 'bfloat16', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16384, 384], 'dtype': 'bfloat16', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [54], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 384]}, {'kind': 'shape', 'name': None, 'dims': [0, 0, 4, 4]}, {'kind': 'shape', 'name': None, 'dims': [32, 3456, 512]}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 384, 9]}, {'kind': 'shape', 'name': None, 'dims': [98304, 64, 9]}, {'kind': 'shape', 'name': None, 'dims': [98304, 64, 9]}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 54]}, {'kind': 'shape', 'name': None, 'dims': [-1, 9, 1]}, {'kind': 'shape', 'name': None, 'dims': [98304, 9, 1]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5]

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
