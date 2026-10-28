"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_3a3508f03d7a
Shape hash: a87f561d
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([401408, 128]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(384):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg0_1, arg1_1, buf0)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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

@triton_heuristics.persistent_reduction(size_hints={'x': 524288, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), '3t', 'c3tfhvepmholxyfdqnrgybddsxrvxfedkyrcer3bnlvi3c5v7u5p.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_clone_convert_element_type_index_mul_permute_sub_sum_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 3, 'num_reduction': 2, 'autotune_hints': (), 'tiling_scores': {'x': 1606080, 'r0_': 1438646784}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_clone_convert_element_type_index_mul_permute_sub_sum_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK: tl.constexpr):
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
    x1 = xindex // 56 % 56
    x0 = xindex % 56
    r0_3 = r0_index
    x2 = xindex // 3136
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + tl.squeeze(x1, 1), None, eviction_policy='evict_last')[:, None]
    tmp6 = tl.load(in_ptr0 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None]
    tmp13 = tl.load(in_ptr2 + tl.squeeze(r0_3, 0), None, eviction_policy='evict_last')[None, :]
    tmp18 = tl.load(in_ptr3 + (r0_3 + 128 * x4), None, eviction_policy='evict_first')
    tmp23 = tl.load(in_ptr4 + tl.squeeze(x4, 1), None, eviction_policy='evict_last')[:, None]
    tmp1 = tl.full([1, 1], 56, tl.int32).to(tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 56), 'index out of bounds: 0 <= tmp4 < 56')
    tmp7 = tmp6 + tmp1
    tmp8 = tmp6 < 0
    tmp9 = tl.where(tmp8, tmp7, tmp6)
    tl.device_assert((0 <= tmp9) & (tmp9 < 56), 'index out of bounds: 0 <= tmp9 < 56')
    tmp11 = tl.load(in_ptr1 + (r0_3 + 128 * (tmp9 % 7) + 896 * (tmp4 % 7) + 6272 * (tmp9 // 7 % 8) + 50176 * (tmp4 // 7 % 8) + 401408 * x2), None, eviction_policy='evict_first').to(tl.float32)
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp19 = tmp14 * tmp18
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp24 = tl.full([1, 1], 128.0, tl.float32)
    tmp25 = tmp14 * tmp24
    tmp26 = tmp25 - tmp17
    tmp27 = tmp18 * tmp22
    tmp28 = tmp26 - tmp27
    tmp29 = tmp23 * tmp28
    tmp30 = tmp12 * tmp18
    tl.store(out_ptr0 + (r0_3 + 128 * x4), tmp12, None)
    tl.store(out_ptr3 + (r0_3 + 128 * x4), tmp29, None)
    tl.store(out_ptr4 + (r0_3 + 128 * x4), tmp30, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((401408, 384), (384, 128)), ((384, 1), (128, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((401408, 128), (128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1, arg4_1, arg5_1), ((56,), (128,), (128, 56, 56, 128), (128, 56, 56, 1)), ((1,), (1,), (401408, 7168, 128, 1), (3136, 56, 1, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        buf1 = empty_strided_cuda((128, 56, 56, 128), (401408, 7168, 128, 1), torch.float32)
        buf4 = empty_strided_cuda((128, 56, 56, 128), (401408, 7168, 128, 1), torch.float32)
        buf5 = empty_strided_cuda((128, 56, 56, 128), (401408, 7168, 128, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_clone_convert_element_type_index_mul_permute_sub_sum_view_1.run(arg2_1, buf0, arg3_1, arg4_1, arg5_1, buf1, buf4, buf5, 401408, 128, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf0
    return (buf1, buf4, buf5)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [401408, 384], 'dtype': 'bfloat16', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [56], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 56}, 'generator': {'kind': 'index', 'low': 0, 'high': 56}}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 56, 56, 128], 'dtype': 'float32', 'stride': [401408, 7168, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 56, 56, 1], 'dtype': 'float32', 'stride': [3136, 56, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8192, 49, 128]}, {'kind': 'shape', 'name': None, 'dims': [8192, 7, 7, 128]}, {'kind': 'shape', 'name': None, 'dims': [128, 8, 8, 7, 7, 128]}, {'kind': 'shape', 'name': None, 'dims': [128, 56, 56, 128]}]}
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
