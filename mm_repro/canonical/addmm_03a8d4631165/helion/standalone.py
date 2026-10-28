"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_03a8d4631165
Shape hash: dfb4d847
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op6_op7(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4, 1000]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = tmp4.to(torch.float32)
        tmp6 = hl.full([], 1.0, dtype=torch.float32)
        tmp7 = tmp6 * tmp5
        tmp8 = tmp6 * acc
        tmp9 = tmp7.reshape([tile_x, 1]) + tmp8.permute(1, 0)
        out0[tile_y, tile_x] = tmp9.permute(1, 0).to(torch.bfloat16)

def call_helion_op6_op7(buf4: torch.Tensor, buf5: torch.Tensor, arg3_1: torch.Tensor, buf7: torch.Tensor) -> None:
    helion_op6_op7(reinterpret_tensor(buf4, (4, 1024), (1024, 1), 0), reinterpret_tensor(buf5, (1024, 1000), (1, 1024), 0), arg3_1, buf7)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 1024, 'r0_': 256}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'p5', 'cp5lm6jr6apsa4mk6cp3zukxdtklcuyjait5wm5bo3xugrkagexd.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_convert_element_type_var_mean_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 16384, 'r0_': 401408}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_convert_element_type_var_mean_0(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1024
    r0_numel = 196
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index % 49
    r0_2 = r0_index // 49
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 49 * x0 + 50176 * r0_2), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(r0_mask & xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tl.full([1, 1], 196, tl.int32).to(tl.float32)
    tmp10 = tmp8 / tmp9
    tmp11 = tmp2 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + x0, tmp10, xmask)
    tl.store(out_ptr1 + x0, tmp16, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(size_hints={'x': 4096, 'r0_': 64}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'iu', 'ciuux5yvdmmj5h74krkkt3ilml64sb7katwk3kmieiin64tz4tey.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_convert_element_type_mean_mul_relu_rsqrt_sub_unsqueeze_var_mean_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 32768, 'r0_': 401408}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_convert_element_type_mean_mul_relu_rsqrt_sub_unsqueeze_var_mean_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 4096
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
    r0_2 = r0_index
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r0_2 + 49 * x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None]
    tmp4 = tl.load(in_ptr2 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None]
    tmp11 = tl.load(in_ptr3 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None]
    tmp13 = tl.load(in_ptr4 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None]
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tl.full([1, 1], 196.0, tl.float32)
    tmp6 = tmp4 / tmp5
    tmp7 = tl.full([1, 1], 1e-05, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1, 1], 0, tl.int32)
    tmp17 = tl.maximum(tmp16, tmp15, tl.PropagateNan.ALL)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(r0_mask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp23 = tl.full([1, 1], 49.0, tl.float32)
    tmp24 = tmp22 / tmp23
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr1 + x3, tmp25, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 1048576}, filename=os.path.join(cache_dir(), '7x', 'c7xbfrb5sn57nci2rvhtt27nvcwx6ma62r4qaxbrra55hihi2cxf.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 8192000}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1024000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, None)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4, 1024, 7, 7), (50176, 49, 7, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        buf1 = empty_strided_cuda((1, 1024, 1, 1), (1024, 1, 1024, 1024), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_convert_element_type_var_mean_0.run(arg0_1, buf0, buf1, 1024, 196, stream=raw_stream0)
        assert_size_stride_grouped((arg1_1, arg2_1), ((1024,), (1024,)), ((1,), (1,)), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf4 = empty_strided_cuda((4, 1024, 1, 1), (1024, 1, 4096, 4096), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_convert_element_type_mean_mul_relu_rsqrt_sub_unsqueeze_var_mean_1.run(arg0_1, buf0, buf1, arg1_1, arg2_1, buf4, 4096, 49, stream=raw_stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del buf0
        del buf1
        assert_size_stride(arg4_1, (1000, 1024), (1024, 1), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        buf5 = empty_strided_cuda((1000, 1024), (1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_2.run(arg4_1, buf5, 1024000, stream=raw_stream0)
        del arg4_1
        assert_size_stride(arg3_1, (1000,), (1,), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf6 = empty_strided_cuda((4, 1000), (1000, 1), torch.bfloat16)
        buf7 = buf6
        del buf6
        raw_stream0 = get_raw_stream(0)
        call_helion_op6_op7(buf4, buf5, arg3_1, buf7)
        del arg3_1
        del buf4
        del buf5
    return (buf7,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4, 1024, 7, 7], 'dtype': 'bfloat16', 'stride': [50176, 49, 7, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000, 1024], 'dtype': 'float32', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [4, 1024]}]}
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
