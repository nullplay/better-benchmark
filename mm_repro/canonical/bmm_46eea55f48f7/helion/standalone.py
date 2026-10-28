"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_46eea55f48f7
Shape hash: 91022e12
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([32768, 56, 56]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(32):
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

@triton_heuristics.persistent_reduction(size_hints={'x': 2097152, 'r0_': 64}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'id', 'cid5cqzwkbeyc3uffzughmxixfjv3qmnih7zqzwfgyi5ix4pq7dx.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_clone_convert_element_type_div_exp_index_permute_prepare_softmax_online_slice_sub_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 3, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 25690112, 'r0_': 472075016}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_clone_convert_element_type_div_exp_index_permute_prepare_softmax_online_slice_sub_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1605632
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
    x0 = xindex % 49
    x4 = xindex // 49
    x1 = xindex // 49 % 4
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_3 + 56 * x0 + 3136 * x4), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_3 + 49 * x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 169, tl.int32).to(tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 169) | ~r0_mask, 'index out of bounds: 0 <= tmp6 < 169')
    tmp8 = tl.load(in_ptr2 + (x1 + 4 * tmp6), r0_mask, eviction_policy='evict_last')
    tmp9 = tmp1 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = tl.where(r0_mask, tmp12, float('-inf'))
    tmp15 = triton_helpers.fmax2(tmp14, 1)[:, None].to(tl.float32)
    tmp16 = tmp10 - tmp15
    tmp17 = libdevice.exp(tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None].to(tl.float32)
    tmp22 = tmp9 - tmp15
    tmp23 = libdevice.exp(tmp22)
    tmp24 = tmp23 / tmp21
    tmp25 = tmp24.to(tl.float32)
    tl.store(out_ptr2 + (r0_3 + 49 * x0 + 2432 * x4), tmp25, r0_mask)
    tl.store(out_ptr0 + x5, tmp15, None)
    tl.store(out_ptr1 + x5, tmp21, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 134217728}, filename=os.path.join(cache_dir(), 'qj', 'cqjlrxskemhrjih22u3cextz6ow6nwrawidk7jn66us7norppnhc.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_clone_constant_pad_nd_convert_element_type_div_exp_index_permute_slice_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 568393728}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_clone_constant_pad_nd_convert_element_type_div_exp_index_permute_slice_sub_unsqueeze_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 102760448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 56 % 56
    x0 = xindex % 56
    x2 = xindex // 3136
    x3 = xindex
    tmp0 = x1.to(tl.int32)
    tmp1 = tl.full([1], 49, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0.to(tl.int32)
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + 49 * x1 + 2432 * x2), tmp5, other=0.0).to(tl.float32)
    tl.store(out_ptr0 + x3, tmp6, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((32768, 56, 32), (32768, 32, 56)), ((1792, 32, 1), (1792, 56, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((32768, 56, 56), (3136, 56, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1), ((49, 49), (169, 4)), ((49, 1), (4, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((8192, 4, 49, 1), (196, 49, 1, 1), torch.float32)
        buf2 = empty_strided_cuda((8192, 4, 49, 1), (196, 49, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((8192, 4, 49, 49), (9728, 2432, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_convert_element_type_div_exp_index_permute_prepare_softmax_online_slice_sub_unsqueeze_view_1.run(buf0, arg2_1, arg3_1, buf1, buf2, buf3, 1605632, 49, stream=raw_stream0)
        del arg2_1
        del arg3_1
        buf4 = buf0
        del buf0
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_constant_pad_nd_convert_element_type_div_exp_index_permute_slice_sub_unsqueeze_view_2.run(buf3, buf4, 102760448, stream=raw_stream0)
    return (buf1, buf2, buf4, reinterpret_tensor(buf3, (32768, 49, 49), (2432, 1, 49), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [32768, 56, 32], 'dtype': 'bfloat16', 'stride': [1792, 32, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32768, 32, 56], 'dtype': 'bfloat16', 'stride': [1792, 56, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [49, 49], 'dtype': 'int64', 'stride': [49, 1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 169}, 'generator': {'kind': 'index', 'low': 0, 'high': 169}}, {'kind': 'tensor', 'name': None, 'shape': [169, 4], 'dtype': 'float32', 'stride': [4, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8192, 4, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [49, 49, -1]}, {'kind': 'shape', 'name': None, 'dims': [8192, 4, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [32768, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [0, 7, 0, 7, 0, 0]}]}
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
