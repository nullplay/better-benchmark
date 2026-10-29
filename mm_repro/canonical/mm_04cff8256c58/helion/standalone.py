"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_04cff8256c58
Shape hash: 02068b68
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1000, 2048]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
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

@triton_heuristics.persistent_reduction(size_hints={'x': 16384, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'wl', 'cwl73hknf6x3atz34q2ocdmyxbpev766avncyce3lv7miaw74vt5.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_cat_convert_element_type_mean_mul_neg_permute_pow_rsqrt_slice_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 8, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 0, 'r0_': 20992768}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_cat_convert_element_type_mean_mul_neg_permute_pow_rsqrt_slice_view_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 16000
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    x3 = xindex // 16
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128 * x0), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp45 = tl.load(in_ptr2 + (r0_1 + 128 * x3), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp47 = tl.load(in_ptr1 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :].to(tl.float32)
    tmp56 = tl.load(in_ptr3 + (r0_1 + 128 * x3), xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = r0_1.to(tl.int32)
    tmp8 = tl.full([1, 1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = r0_1.to(tl.int64)
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tl.full([1, 1], 64, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = tl.load(in_ptr1 + tl.broadcast_to(64 + r0_1, [XBLOCK, R0_BLOCK]), tmp13 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (64 + 128 * x0 + r0_1), tmp13 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.full([1, 1], 128.0, tl.float32)
    tmp18 = tmp6 / tmp17
    tmp19 = tl.full([1, 1], 1e-06, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp14 * tmp23
    tmp25 = -tmp24
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp13, tmp25, tmp26)
    tmp28 = tmp7 >= tmp12
    tmp29 = tl.full([1, 1], 128, tl.int64)
    tmp30 = tmp7 < tmp29
    tmp31 = tl.load(in_ptr1 + tl.broadcast_to(-64 + r0_1, [XBLOCK, R0_BLOCK]), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tl.load(in_ptr0 + (128 * x0 + (-64 + r0_1)), tmp28 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.full([1, 1], 128.0, tl.float32)
    tmp35 = tmp6 / tmp34
    tmp36 = tl.full([1, 1], 1e-06, tl.float32)
    tmp37 = tmp35 + tmp36
    tmp38 = libdevice.rsqrt(tmp37)
    tmp39 = tmp33 * tmp38
    tmp40 = tmp39.to(tl.float32)
    tmp41 = tmp31 * tmp40
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp28, tmp41, tmp42)
    tmp44 = tl.where(tmp13, tmp27, tmp43)
    tmp46 = tmp44 * tmp45
    tmp48 = tl.full([1, 1], 128.0, tl.float32)
    tmp49 = tmp6 / tmp48
    tmp50 = tl.full([1, 1], 1e-06, tl.float32)
    tmp51 = tmp49 + tmp50
    tmp52 = libdevice.rsqrt(tmp51)
    tmp53 = tmp1 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp47 * tmp54
    tmp57 = tmp55 * tmp56
    tmp58 = tmp57 + tmp46
    tl.store(in_out_ptr0 + (r0_1 + 128 * x0), tmp58, xmask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((1000, 1024), (1024, 2048)), ((1024, 1), (1, 1024)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((1000, 2048), (2048, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg4_1, arg3_1), ((128,), (1, 1, 1000, 128), (1, 1, 1000, 128)), ((1,), (128000, 128000, 128, 1), (128000, 128000, 128, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf2 = empty_strided_cuda((1, 16, 1000, 128), (2048000, 128, 2048, 1), torch.bfloat16)
        buf3 = buf2
        del buf2
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_cat_convert_element_type_mean_mul_neg_permute_pow_rsqrt_slice_view_1.run(buf3, buf0, arg2_1, arg4_1, arg3_1, 16000, 128, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del buf0
    return (buf3,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1024, 2048], 'dtype': 'bfloat16', 'stride': [1, 1024], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000, 1024], 'dtype': 'bfloat16', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 1, 1000, 128], 'dtype': 'bfloat16', 'stride': [128000, 128000, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 1, 1000, 128], 'dtype': 'bfloat16', 'stride': [128000, 128000, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, 16, 128]}]}
COMPILED_INPUT_ORDER = [1, 0, 2, 3, 4]

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
