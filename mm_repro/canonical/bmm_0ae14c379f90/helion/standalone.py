"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_0ae14c379f90
Shape hash: f6ba3090
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([768, 49, 49]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op2(buf1: torch.Tensor, arg1_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op2(reinterpret_tensor(buf1, (768, 49, 128), (6272, 128, 1), 0), arg1_1, buf2)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op5(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([768, 128, 49]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(49):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op5(arg7_1: torch.Tensor, buf4: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op5(arg7_1, reinterpret_tensor(buf4, (768, 49, 49), (2401, 49, 1), 0), buf5)
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

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'mh', 'cmh64menepyzglcixszfm7y4fehcarcbb2fv2pno6trr27pure22.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_permute_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 28901376}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_permute_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 128
    x1 = xindex // 128 % 49
    x2 = xindex // 6272 % 6
    x3 = xindex // 37632
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 128 * x2 + 768 * x1 + 37632 * x3), None).to(tl.float32)
    tl.store(out_ptr0 + x4, tmp0, None)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 65536, 'r0_': 64}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'w5', 'cw5p6u774shwtc26cikyhdjbao3iff5gg2htuc35uparqv2nf26p.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i1', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_convert_element_type_div_exp_fma_mul_neg_slice_sub_sum_view_where_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 489216, 'r0_': 14751744}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_convert_element_type_div_exp_fma_mul_neg_slice_sub_sum_view_where_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 37632
    r0_numel = 49
    R0_BLOCK: tl.constexpr = 64
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
    x3 = xindex
    r0_2 = r0_index
    x0 = xindex % 49
    x1 = xindex // 49
    tmp0 = tl.load(in_ptr0 + tl.squeeze(x3, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None].to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 56 * x0 + 3136 * x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + tl.squeeze(x3, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None]
    tmp9 = tl.load(in_ptr3 + tl.squeeze(x3, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None]
    tmp13 = tl.load(in_ptr4 + tl.squeeze(x3, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None]
    tmp15 = tl.load(in_out_ptr0 + (r0_2 + 49 * x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 - tmp3
    tmp5 = tl.full([1, 1], 0.08838834764831845, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tmp1 * tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 - tmp9
    tmp11 = tl.where(tmp0, tmp6, tmp10)
    tmp12 = libdevice.exp(tmp11)
    tmp14 = tmp12 / tmp13
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp14
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask & xmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = -tmp14
    tmp23 = tl.fma(tmp22, tmp21, tmp17)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 * tmp5
    tl.store(in_out_ptr0 + (r0_2 + 49 * x3), tmp25, r0_mask & xmask)
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
    assert_size_stride(arg0_1, (128, 768, 7, 7), (37632, 1, 5376, 768), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf1 = empty_strided_cuda((128, 6, 49, 128), (37632, 6272, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_view_0.run(arg0_1, buf1, 4816896, stream=raw_stream0)
        del arg0_1
        assert_size_stride(arg1_1, (768, 128, 49), (6272, 1, 128), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf2 = empty_strided_cuda((768, 49, 49), (2401, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2(buf1, arg1_1, buf2)
        del arg1_1
        del buf1
        assert_size_stride_grouped((arg5_1, arg2_1, arg3_1, arg4_1, arg6_1), ((128, 6, 49, 1), (768, 56, 56), (128, 6, 49, 1), (128, 6, 49, 1), (128, 6, 49, 1)), ((294, 49, 1, 1), (3136, 56, 1), (294, 49, 1, 1), (294, 49, 1, 1), (294, 49, 1, 1)), 'input')
        arg5_1 = copy_if_misaligned(arg5_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        buf4 = reinterpret_tensor(buf2, (128, 6, 49, 49), (14406, 2401, 49, 1), 0)
        del buf2
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_convert_element_type_div_exp_fma_mul_neg_slice_sub_sum_view_where_2.run(buf4, arg5_1, arg2_1, arg3_1, arg4_1, arg6_1, 37632, 49, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        assert_size_stride(arg7_1, (768, 128, 49), (6272, 1, 128), 'input')
        arg7_1 = copy_if_misaligned(arg7_1)
        buf5 = empty_strided_cuda((768, 128, 49), (6272, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op5(arg7_1, buf4, buf5)
        del arg7_1
    return (reinterpret_tensor(buf4, (768, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf5, (128, 6, 49, 128), (37632, 6272, 1, 49), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [128, 768, 7, 7], 'dtype': 'bfloat16', 'stride': [37632, 1, 5376, 768], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 128, 49], 'dtype': 'bfloat16', 'stride': [6272, 1, 128], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 56, 56], 'dtype': 'bfloat16', 'stride': [3136, 56, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 6, 49, 1], 'dtype': 'float32', 'stride': [294, 49, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 6, 49, 1], 'dtype': 'float32', 'stride': [294, 49, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 6, 49, 1], 'dtype': 'bool', 'stride': [294, 49, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 6, 49, 1], 'dtype': 'float32', 'stride': [294, 49, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 128, 49], 'dtype': 'bfloat16', 'stride': [6272, 1, 128], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [128, 6, 128, 49]}, {'kind': 'shape', 'name': None, 'dims': [768, 49, 128]}, {'kind': 'shape', 'name': None, 'dims': [128, 6, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [128, 6, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [768, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [128, 6, 128, 49]}]}
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
