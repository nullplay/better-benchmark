"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_22f44887e68d
Shape hash: 5a9245b3
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([8192, 49, 49]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(32):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(arg0_1, arg1_1, buf1)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op5(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([8192, 32, 49]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(49):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op5(arg8_1: torch.Tensor, buf4: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op5(arg8_1, reinterpret_tensor(buf4, (8192, 49, 49), (2401, 49, 1), 0), buf5)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 524288, 'r0_': 64}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'l3', 'cl3laf3u43e3a6jkytjxwkjsv54ojes7nq7qvtraufabmwvvcfqs.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_clone_convert_element_type_div_exp_fma_index_mul_neg_permute_slice_sub_sum_unsqueeze_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 6, 'num_store': 2, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 3211264, 'r0_': 314761496}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_clone_convert_element_type_div_exp_fma_index_mul_neg_permute_slice_sub_sum_unsqueeze_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 401408
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
    x1 = xindex // 49 % 16
    x2 = xindex // 784
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_3 + 56 * x0 + 3136 * x4), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_3 + 49 * x0), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (r0_3 + 49 * x0 + 2401 * (x2 % 4)), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + tl.squeeze(x5, 1), None, eviction_policy='evict_last')[:, None]
    tmp15 = tl.load(in_ptr5 + tl.squeeze(x5, 1), None, eviction_policy='evict_last')[:, None]
    tmp17 = tl.load(in_ptr6 + (r0_3 + 49 * x5), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tl.full([1, 1], 169, tl.int32).to(tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 169) | ~r0_mask, 'index out of bounds: 0 <= tmp6 < 169')
    tmp8 = tl.load(in_ptr2 + (x1 + 16 * tmp6), r0_mask, eviction_policy='evict_last')
    tmp9 = tmp1 + tmp8
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 - tmp12
    tmp14 = libdevice.exp(tmp13)
    tmp16 = tmp14 / tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp16
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None].to(tl.float32)
    tmp24 = -tmp16
    tmp25 = tl.fma(tmp24, tmp23, tmp19)
    tmp26 = tmp25.to(tl.float32)
    tl.store(out_ptr2 + (r0_3 + 49 * x5), tmp25, r0_mask)
    tl.store(out_ptr3 + (r0_3 + 49 * x5), tmp26, r0_mask)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((8192, 49, 32), (8192, 32, 49)), ((1568, 32, 1), (1600, 1, 32)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((8192, 49, 49), (2401, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(arg0_1, arg1_1, buf1)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1), ((8192, 56, 56), (49, 49), (169, 16), (4, 49, 49), (512, 16, 49, 1), (512, 16, 49, 1)), ((3136, 56, 1), (49, 1), (16, 1), (2401, 49, 1), (784, 49, 1, 1), (784, 49, 1, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        arg7_1 = copy_if_misaligned(arg7_1)
        buf3 = empty_strided_cuda((512, 16, 49, 49), (38416, 2401, 49, 1), torch.float32)
        buf4 = empty_strided_cuda((512, 16, 49, 49), (38416, 2401, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_clone_convert_element_type_div_exp_fma_index_mul_neg_permute_slice_sub_sum_unsqueeze_view_1.run(arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, buf1, buf3, buf4, 401408, 49, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf1
        assert_size_stride(arg8_1, (8192, 32, 49), (1600, 1, 32), 'input')
        arg8_1 = copy_if_misaligned(arg8_1)
        buf5 = empty_strided_cuda((8192, 32, 49), (1568, 49, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op5(arg8_1, buf4, buf5)
        del arg8_1
    return (buf3, reinterpret_tensor(buf4, (8192, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf5, (512, 16, 49, 32), (25088, 1568, 1, 49), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [8192, 49, 32], 'dtype': 'bfloat16', 'stride': [1568, 32, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 32, 49], 'dtype': 'bfloat16', 'stride': [1600, 1, 32], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 56, 56], 'dtype': 'bfloat16', 'stride': [3136, 56, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [49, 49], 'dtype': 'int64', 'stride': [49, 1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 169}, 'generator': {'kind': 'index', 'low': 0, 'high': 169}}, {'kind': 'tensor', 'name': None, 'shape': [169, 16], 'dtype': 'float32', 'stride': [16, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4, 49, 49], 'dtype': 'float32', 'stride': [2401, 49, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 16, 49, 1], 'dtype': 'float32', 'stride': [784, 49, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 16, 49, 1], 'dtype': 'float32', 'stride': [784, 49, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 32, 49], 'dtype': 'bfloat16', 'stride': [1600, 1, 32], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [49, 49, -1]}, {'kind': 'shape', 'name': None, 'dims': [-1, 4, 16, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [-1, 16, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [8192, 49, 49]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 32, 49]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8]

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
