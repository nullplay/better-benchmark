"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_bmm_fe4891df3e8d
Shape hash: 37923383
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([32768, 128]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg2_1: torch.Tensor, arg0_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg2_1, reinterpret_tensor(arg0_1, (128, 128), (1, 128), 0), buf0)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([32768, 128]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(arg5_1: torch.Tensor, arg3_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(arg5_1, reinterpret_tensor(arg3_1, (128, 128), (1, 128), 0), buf1)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op4(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([1024, 128, 128]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(32):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op4(buf2: torch.Tensor, buf3: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op4(reinterpret_tensor(buf2, (1024, 128, 32), (4096, 32, 1), 0), reinterpret_tensor(buf3, (1024, 32, 128), (4096, 128, 1), 0), buf4)
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

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), 'vc', 'cvck3cui2vqwkyv37j6ftgobv5ngysr7cop5dbmwrebcs55juzoo.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_clone_mul_permute_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 25166080}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_clone_mul_permute_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 32
    x2 = xindex // 4096 % 4
    x1 = xindex // 32 % 128
    x3 = xindex // 16384
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0 + 32 * x2 + 128 * x1 + 16384 * x3), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tmp6 = tl.full([1], 0.4204482076268573, tl.float32)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + x4, tmp7, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'y': 32768, 'x': 128}, tile_hint=TileHint.DEFAULT, filename=os.path.join(cache_dir(), '45', 'c45hsqd46imciwbff4frg5t3utk7ln3ln5qqxnvhk7munzopajlp.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_addmm_clone_mul_permute_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 8388608, 'x': 16777472}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_clone_mul_permute_view_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 32768
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex % 128
    x2 = xindex
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + tl.squeeze(y0, 1), None, eviction_policy='evict_last')[:, None].to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (y0 + 128 * x2 + 16384 * y1), xmask).to(tl.float32)
    tmp1 = tl.full([1, 1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tmp6 = tl.full([1, 1], 0.4204482076268573, tl.float32)
    tmp7 = tmp5 * tmp6
    tl.store(out_ptr0 + (x2 + 128 * y3), tmp7, xmask)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 131072, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'fy', 'cfyk3glqx372whnxrbmedzvnjycgekhnra4w7bcmi6emmdhtxrkc.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_any_convert_element_type_div_eq_exp_full_logical_not_prepare_softmax_online_sub_view_where_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 5, 'autotune_hints': (), 'tiling_scores': {'x': 0, 'r0_': 100663296}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_any_convert_element_type_div_eq_exp_full_logical_not_prepare_softmax_online_sub_view_where_4(in_out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 131072
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 128 * x0), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = tl.full([1, 1], float('-inf'), tl.float32)
    tmp2 = tmp0 == tmp1
    tmp3 = tmp2 == 0
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tmp4 != 0
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = triton_helpers.any(tmp6, 1)[:, None].to(tl.int1)
    tmp9 = tmp0.to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp14 = triton_helpers.fmax2(tmp12, 1)[:, None].to(tl.float32)
    tmp15 = tmp10 - tmp14
    tmp16 = libdevice.exp(tmp15)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tmp20 = tmp8 == 0
    tmp21 = tmp9 - tmp14
    tmp22 = libdevice.exp(tmp21)
    tmp23 = tmp22 / tmp19
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.full([1, 1], 0.0, tl.float32)
    tmp26 = tl.where(tmp20, tmp25, tmp24)
    tl.store(in_out_ptr0 + (r0_1 + 128 * x0), tmp26, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride_grouped((arg2_1, arg0_1), ((32768, 128), (128, 128)), ((128, 1), (128, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((32768, 128), (128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg2_1, arg0_1, buf0)
        del arg0_1
        del arg2_1
        assert_size_stride_grouped((arg5_1, arg3_1), ((32768, 128), (128, 128)), ((128, 1), (128, 1)), 'input')
        arg5_1 = copy_if_misaligned(arg5_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((32768, 128), (128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(arg5_1, arg3_1, buf1)
        del arg3_1
        del arg5_1
        assert_size_stride(arg1_1, (128,), (1,), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf2 = empty_strided_cuda((256, 4, 128, 32), (16384, 4096, 32, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_mul_permute_view_1.run(arg1_1, buf0, buf2, 4194304, stream=raw_stream0)
        del arg1_1
        assert_size_stride(arg4_1, (128,), (1,), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        buf3 = reinterpret_tensor(buf0, (256, 4, 32, 128), (16384, 4096, 128, 1), 0)
        del buf0
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_mul_permute_view_2.run(arg4_1, buf1, buf3, 32768, 128, stream=raw_stream0)
        del arg4_1
        del buf1
        buf4 = empty_strided_cuda((1024, 128, 128), (16384, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op4(buf2, buf3, buf4)
        del buf2
        del buf3
        buf8 = reinterpret_tensor(buf4, (256, 4, 128, 128), (65536, 16384, 128, 1), 0)
        del buf4
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_any_convert_element_type_div_eq_exp_full_logical_not_prepare_softmax_online_sub_view_where_4.run(buf8, 131072, 128, stream=raw_stream0)
    return (reinterpret_tensor(buf8, (1024, 128, 128), (16384, 128, 1), 0),)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [128, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32768, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32768, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'shape', 'name': None, 'dims': [256, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [256, 128, -1, 32]}, {'kind': 'shape', 'name': None, 'dims': [256, 4, 128, 32]}, {'kind': 'shape', 'name': None, 'dims': [1024, 128, 32]}, {'kind': 'shape', 'name': None, 'dims': [256, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [256, 128, -1, 32]}, {'kind': 'shape', 'name': None, 'dims': [256, 4, 32, 128]}, {'kind': 'shape', 'name': None, 'dims': [1024, 32, 128]}, {'kind': 'shape', 'name': None, 'dims': [256, 4, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [256, 4, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [256, 4, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [1024, 128, 128]}], 'alias_group_nbytes': [8388608]}
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
