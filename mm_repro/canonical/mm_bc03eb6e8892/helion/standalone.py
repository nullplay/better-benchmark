"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_bc03eb6e8892
Shape hash: fe02163a
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 54]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(384):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(arg2_1: torch.Tensor, buf0: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(arg2_1, reinterpret_tensor(buf0, (384, 54), (1, 384), 0), buf1)
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

@triton_heuristics.pointwise(size_hints={'x': 32768}, filename=os.path.join(cache_dir(), 'wm', 'cwmy5bidmtv4xdah67rusndpzql3j3vc3mtcsqahc2bozvmflzdp.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 165888}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 20736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, xmask)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 131072, 'r0_': 16}, reduction_hint=ReductionHint.DEFAULT, filename=os.path.join(cache_dir(), 'dl', 'cdlanp62sxt2b7wo6vnniu3iwij36k2b4yd7je7shva25j7j3riq.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_convert_element_type_div_exp_expand_prepare_softmax_online_sub_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 3, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 1572864, 'r0_': 5308632}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_convert_element_type_div_exp_expand_prepare_softmax_online_sub_view_2(in_out_ptr0, in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
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
    tmp1 = tl.load(in_ptr0 + (r0_1 + 9 * x0) % 54, r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask, tmp7, float('-inf'))
    tmp10 = triton_helpers.fmax2(tmp9, 1)[:, None].to(tl.float32)
    tmp11 = tmp5 - tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tmp17 = tmp4 - tmp10
    tmp18 = libdevice.exp(tmp17)
    tmp19 = tmp18 / tmp16
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 9 * x0), tmp20, r0_mask)
    tl.store(out_ptr0 + x0, tmp10, None)
    tl.store(out_ptr1 + x0, tmp16, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg1_1, (54, 384), (384, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((54, 384), (384, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg1_1, buf0, 20736, stream=raw_stream0)
        del arg1_1
        assert_size_stride(arg2_1, (16384, 384), (384, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((16384, 54), (54, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(arg2_1, buf0, buf1)
        del arg2_1
        del buf0
        assert_size_stride(arg0_1, (54,), (1,), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf2 = empty_strided_cuda((98304, 1, 1), (1, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((98304, 1, 1), (1, 1, 1), torch.float32)
        buf4 = reinterpret_tensor(buf1, (98304, 9, 1), (9, 1, 1), 0)
        del buf1
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_convert_element_type_div_exp_expand_prepare_softmax_online_sub_view_2.run(buf4, arg0_1, buf2, buf3, 98304, 9, stream=raw_stream0)
        del arg0_1
    return (buf2, buf3, buf4)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [54], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [54, 384], 'dtype': 'float32', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16384, 384], 'dtype': 'bfloat16', 'stride': [384, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 54]}, {'kind': 'shape', 'name': None, 'dims': [-1, 9, 1]}, {'kind': 'shape', 'name': None, 'dims': [98304, 9, 1]}]}
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
