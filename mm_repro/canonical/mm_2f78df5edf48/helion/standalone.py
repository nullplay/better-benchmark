"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_2f78df5edf48
Shape hash: 3fd186e8
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg2_1: torch.Tensor, arg1_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg2_1, reinterpret_tensor(arg1_1, (4096, 4096), (1, 4096), 0), buf0)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op4(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op4(arg4_1: torch.Tensor, arg3_1: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op4(arg4_1, reinterpret_tensor(arg3_1, (4096, 4096), (1, 4096), 0), buf4)
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
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 524288}, filename=os.path.join(cache_dir(), 'xl', 'cxldewqfbxzgugazngowxm35wtgotqfdbaplwgij2obd4iuljiz5.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr3': '*fp32', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_permute_repeat_slice_split_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 14811136}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_permute_repeat_slice_split_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr3, out_ptr5, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 256
    x1 = xindex // 256 % 128
    x2 = xindex // 32768
    x4 = xindex
    tmp0 = x0.to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1], 64, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (256 * x2 + 4096 * x1 + x0), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp8 = tl.load(in_ptr1 + (32 + 64 * x1 + x0 // 2 % 32), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = (x0 % 2).to(tl.int32)
    tmp11 = tl.full([1], 0, tl.int64)
    tmp12 = tmp10 >= tmp11
    tmp13 = (x0 % 2).to(tl.int64)
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.full([1], 1, tl.int64)
    tmp16 = tmp14 < tmp15
    tmp17 = tmp16 & tmp6
    tmp18 = tl.load(in_ptr0 + (1 + 2 * (x0 // 2 % 32) + 256 * x2 + 4096 * x1), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = -tmp18
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp17, tmp19, tmp20)
    tmp22 = tmp10 >= tmp15
    tmp23 = tl.full([1], 2, tl.int64)
    tmp24 = tmp10 < tmp23
    tmp25 = tmp22 & tmp6
    tmp26 = tl.load(in_ptr0 + (2 * (x0 // 2 % 32) + 256 * x2 + 4096 * x1), tmp25, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tl.where(tmp16, tmp21, tmp26)
    tmp28 = tl.load(in_ptr1 + (64 * x1 + x0 // 2 % 32), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp9 + tmp29
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp6, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tl.full([1], 256, tl.int64)
    tmp35 = tmp0 < tmp34
    tmp36 = tl.load(in_ptr0 + (64 + 256 * x2 + 4096 * x1 + (-64 + x0)), tmp33, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp37 = tl.where(tmp6, tmp32, tmp36)
    tmp38 = tl.load(in_ptr2 + (256 * x2 + 4096 * x1 + x0), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tmp38 * tmp8
    tmp40 = tl.load(in_ptr2 + (1 + 2 * (x0 // 2 % 32) + 256 * x2 + 4096 * x1), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp41 = -tmp40
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp17, tmp41, tmp42)
    tmp44 = tl.load(in_ptr2 + (2 * (x0 // 2 % 32) + 256 * x2 + 4096 * x1), tmp25, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp45 = tl.where(tmp16, tmp43, tmp44)
    tmp46 = tmp45 * tmp28
    tmp47 = tmp39 + tmp46
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp6, tmp47, tmp48)
    tmp50 = tl.load(in_ptr2 + (64 + 256 * x2 + 4096 * x1 + (-64 + x0)), tmp33, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp51 = tl.where(tmp6, tmp49, tmp50)
    tmp52 = tmp37.to(tl.float32)
    tmp53 = tmp51.to(tl.float32)
    tl.store(out_ptr3 + (x0 + 256 * x2 + 4096 * x1), tmp52, None)
    tl.store(out_ptr5 + (x0 + 256 * x2 + 4096 * x1), tmp53, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride_grouped((arg2_1, arg1_1), ((128, 4096), (4096, 4096)), ((4096, 1), (4096, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((128, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg2_1, arg1_1, buf0)
        del arg1_1
        del arg2_1
        assert_size_stride_grouped((arg4_1, arg3_1), ((128, 4096), (4096, 4096)), ((4096, 1), (4096, 1)), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf4 = empty_strided_cuda((128, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op4(arg4_1, arg3_1, buf4)
        del arg3_1
        del arg4_1
        assert_size_stride(arg0_1, (2048, 64), (64, 1), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf3 = empty_strided_cuda((16, 128, 256), (256, 4096, 1), torch.float32)
        buf7 = empty_strided_cuda((16, 256, 128), (256, 1, 4096), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_clone_convert_element_type_expand_gather_mul_neg_permute_repeat_slice_split_unsqueeze_view_1.run(buf0, arg0_1, buf4, buf3, buf7, 524288, stream=raw_stream0)
        del arg0_1
        del buf0
        del buf4
    return (buf3, buf7)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [2048, 64], 'dtype': 'bfloat16', 'stride': [64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 4096]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 128, 256]}, {'kind': 'shape', 'name': None, 'dims': [16, 128, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 4096]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 256, 128]}, {'kind': 'shape', 'name': None, 'dims': [16, 256, 128]}], 'alias_group_nbytes': [1048576]}
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
