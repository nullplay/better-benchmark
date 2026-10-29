"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_f08a2f311c4e
Shape hash: 5aee1e44
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 128]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg1_1, arg2_1, buf0)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op6_op7_op8(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 30000]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6 + tmp7
        out0[tile_y, tile_x] = tmp8.to(torch.bfloat16)
        tmp9 = tmp8.to(torch.float32)
        out1[tile_y, tile_x] = tmp9.to(torch.float32)

def call_helion_op6_op7_op8(buf5: torch.Tensor, arg6_1: torch.Tensor, arg5_1: torch.Tensor, buf7: torch.Tensor, buf8: torch.Tensor) -> None:
    helion_op6_op7_op8(reinterpret_tensor(buf5, (4096, 128), (128, 1), 0), arg6_1, arg5_1, buf7, buf8)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 4096, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'p3', 'cp3nrbhp6ntpnvvjjb7mmqikmoz7ksvmhymmy5nhxaapynycjotb.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_addmm_convert_element_type_mul_pow_rsqrt_sub_tanh_var_mean_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 0, 'r0_': 3146496}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_addmm_convert_element_type_mul_pow_rsqrt_sub_tanh_var_mean_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr3, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :].to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (r0_1 + 128 * x0), None, eviction_policy='evict_first').to(tl.float32)
    tmp38 = tl.load(in_ptr2 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :].to(tl.float32)
    tmp41 = tl.load(in_ptr3 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :].to(tl.float32)
    tmp1 = tl.full([1, 1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tmp6 = tl.full([1, 1], 0.5, tl.float32)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp5 * tmp5
    tmp9 = tmp8 * tmp5
    tmp10 = tl.full([1, 1], 0.044715, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp5 + tmp11
    tmp13 = tl.full([1, 1], 0.7978845608028654, tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = libdevice.tanh(tmp14)
    tmp16 = tmp15 + tmp1
    tmp17 = tmp7 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp24 = tl.full([1, 1], 128, tl.int32).to(tl.float32)
    tmp25 = tmp23 / tmp24
    tmp26 = tmp19 - tmp25
    tmp27 = tmp26 * tmp26
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.sum(tmp28, 1)[:, None].to(tl.float32)
    tmp31 = tmp18 - tmp25
    tmp32 = tl.full([1, 1], 128.0, tl.float32)
    tmp33 = tmp30 / tmp32
    tmp34 = tl.full([1, 1], 1e-12, tl.float32)
    tmp35 = tmp33 + tmp34
    tmp36 = libdevice.rsqrt(tmp35)
    tmp37 = tmp31 * tmp36
    tmp39 = tmp38.to(tl.float32)
    tmp40 = tmp37 * tmp39
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp40 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tl.store(out_ptr3 + (r0_1 + 128 * x0), tmp44, None)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride_grouped((arg1_1, arg2_1), ((4096, 4096), (4096, 128)), ((4096, 1), (1, 4096)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf0 = empty_strided_cuda((4096, 128), (128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg1_1, arg2_1, buf0)
        del arg1_1
        del arg2_1
        assert_size_stride_grouped((arg0_1, arg3_1, arg4_1), ((128,), (128,), (128,)), ((1,), (1,), (1,)), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf5 = empty_strided_cuda((8, 512, 128), (65536, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convert_element_type_mul_pow_rsqrt_sub_tanh_var_mean_view_1.run(arg0_1, buf0, arg3_1, arg4_1, buf5, 4096, 128, stream=raw_stream0)
        del arg0_1
        del arg3_1
        del arg4_1
        del buf0
        assert_size_stride_grouped((arg6_1, arg5_1), ((128, 30000), (30000,)), ((1, 128), (1,)), 'input')
        arg6_1 = copy_if_misaligned(arg6_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        buf7 = empty_strided_cuda((4096, 30000), (30000, 1), torch.bfloat16)
        buf8 = empty_strided_cuda((4096, 30000), (30000, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op6_op7_op8(buf5, arg6_1, arg5_1, buf7, buf8)
        del arg5_1
        del arg6_1
        del buf5
    return (reinterpret_tensor(buf7, (8, 512, 30000), (15360000, 30000, 1), 0), buf8)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 128], 'dtype': 'bfloat16', 'stride': [1, 4096], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [30000], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 30000], 'dtype': 'bfloat16', 'stride': [1, 128], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 128]}, {'kind': 'shape', 'name': None, 'dims': [4096, 128]}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 30000]}]}
COMPILED_INPUT_ORDER = [1, 2, 0, 3, 4, 5, 6]

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
