"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_89e0ab13d216
Shape hash: bc26879a
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2_op3(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16, 1000]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1280):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        out0[tile_y, tile_x] = tmp8.permute(1, 0).to(torch.bfloat16)

def call_helion_op2_op3(buf1: torch.Tensor, arg5_1: torch.Tensor, arg6_1: torch.Tensor, buf3: torch.Tensor) -> None:
    helion_op2_op3(reinterpret_tensor(buf1, (16, 1280), (1280, 1), 0), reinterpret_tensor(arg5_1, (1280, 1000), (1, 1280), 0), arg6_1, buf3)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 32768, 'r0_': 64}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'ml', 'cmlq66t2wq4su7chplivjj6i7d2prrlm37d3mcuskm2yvsvmhppr.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_clamp_max_clamp_min_convert_element_type_mean_mul_sub_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 5, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 92160, 'r0_': 2007040}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_clamp_max_clamp_min_convert_element_type_mean_mul_sub_unsqueeze_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 20480
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
    x0 = xindex % 1280
    tmp0 = tl.load(in_ptr0 + (r0_2 + 49 * x3), r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None].to(tl.float32)
    tmp5 = tl.load(in_ptr2 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None].to(tl.float32)
    tmp11 = tl.load(in_ptr3 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None].to(tl.float32)
    tmp14 = tl.load(in_ptr4 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None].to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 - tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.full([1, 1], 1e-05, tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp4 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 * tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp13 + tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tl.full([1, 1], 0.0, tl.float32)
    tmp20 = tl.maximum(tmp18, tmp19, tl.PropagateNan.ALL)
    tmp21 = tl.full([1, 1], 6.0, tl.float32)
    tmp22 = tl.minimum(tmp20, tmp21, tl.PropagateNan.ALL)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
    tmp27 = tl.where(r0_mask, tmp25, 0)
    tmp28 = tl.sum(tmp27, 1)[:, None].to(tl.float32)
    tmp29 = tl.full([1, 1], 49.0, tl.float32)
    tmp30 = tmp28 / tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.store(out_ptr1 + x3, tmp31, None)
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
    assert_size_stride_grouped((arg1_1, arg0_1, arg2_1, arg3_1, arg4_1), ((16, 1280, 7, 7), (1280,), (1280,), (1280,), (1280,)), ((62720, 49, 7, 1), (1,), (1,), (1,), (1,)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf1 = empty_strided_cuda((16, 1280, 1, 1), (1280, 1, 20480, 20480), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_clamp_max_clamp_min_convert_element_type_mean_mul_sub_unsqueeze_0.run(arg1_1, arg0_1, arg2_1, arg3_1, arg4_1, buf1, 20480, 49, stream=raw_stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        assert_size_stride_grouped((arg5_1, arg6_1), ((1000, 1280), (1000,)), ((1280, 1), (1,)), 'input')
        arg5_1 = copy_if_misaligned(arg5_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        buf2 = empty_strided_cuda((16, 1000), (1000, 1), torch.bfloat16)
        buf3 = buf2
        del buf2
        raw_stream0 = get_raw_stream(0)
        call_helion_op2_op3(buf1, arg5_1, arg6_1, buf3)
        del arg5_1
        del arg6_1
        del buf1
    return (buf3,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1280], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 1280, 7, 7], 'dtype': 'bfloat16', 'stride': [62720, 49, 7, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1280], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1280], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1280], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000, 1280], 'dtype': 'bfloat16', 'stride': [1280, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [16, 1280]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6]

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
