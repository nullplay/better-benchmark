"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_1f15df84c407
Shape hash: 1538d528
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([32768, 128]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(256):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(arg2_1: torch.Tensor, buf0: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(arg2_1, reinterpret_tensor(buf0, (256, 128), (1, 256), 0), buf1)
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

@triton_heuristics.pointwise(size_hints={'x': 32768}, filename=os.path.join(cache_dir(), 'of', 'cofk5dd3afcfb37ekmdwpgpdzgxi7qf3tozvkui7tvoxgb3c2amo.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 262144}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 32768
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
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(size_hints={'x': 32768, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'rf', 'crfj3wgvk6gnmeyjhpb33ukmijta4odllzmakmftkr52nmhej4ir.py'), triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_addmm_convert_element_type_erf_mul_permute_rsqrt_sub_var_mean_view_2', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 4, 'num_store': 3, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 524288, 'r0_': 25167360}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_addmm_convert_element_type_erf_mul_permute_rsqrt_sub_var_mean_view_2(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 32768
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
    tmp0 = tl.load(in_ptr0 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :]
    tmp4 = tl.load(in_out_ptr1 + (r0_1 + 128 * x0), None, eviction_policy='evict_first').to(tl.float32)
    tmp36 = tl.load(in_ptr1 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :]
    tmp38 = tl.load(in_ptr2 + tl.squeeze(r0_1, 0), None, eviction_policy='evict_last')[None, :]
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1, 1], 1.0, tl.float32)
    tmp3 = tmp2 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tl.full([1, 1], 0.5, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp11 = tmp7 * tmp10
    tmp12 = libdevice.erf(tmp11)
    tmp13 = tmp12 + tmp2
    tmp14 = tmp9 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp21 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp22 = tl.full([1, 1], 128, tl.int32).to(tl.float32)
    tmp23 = tmp21 / tmp22
    tmp24 = tmp17 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tmp29 = tl.full([1, 1], 128.0, tl.float32)
    tmp30 = tmp28 / tmp29
    tmp31 = tl.full([1, 1], 1e-12, tl.float32)
    tmp32 = tmp30 + tmp31
    tmp33 = libdevice.rsqrt(tmp32)
    tmp34 = tmp16 - tmp23
    tmp35 = tmp34 * tmp33
    tmp37 = tmp35 * tmp36
    tmp39 = tmp37 + tmp38
    tmp40 = tmp39.to(tl.float32)
    tl.store(in_out_ptr0 + x0, tmp33, None)
    tl.store(in_out_ptr1 + (r0_1 + 128 * x0), tmp40, None)
    tl.store(out_ptr0 + x0, tmp23, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg1_1, (128, 256), (256, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((128, 256), (256, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg1_1, buf0, 32768, stream=raw_stream0)
        del arg1_1
        assert_size_stride(arg2_1, (32768, 256), (256, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((32768, 128), (128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(arg2_1, buf0, buf1)
        del arg2_1
        del buf0
        assert_size_stride_grouped((arg0_1, arg3_1, arg4_1), ((128,), (128,), (128,)), ((1,), (1,), (1,)), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf2 = empty_strided_cuda((64, 512, 1), (512, 1, 1), torch.float32)
        buf3 = empty_strided_cuda((64, 512, 1), (512, 1, 32768), torch.float32)
        buf5 = reinterpret_tensor(buf3, (64, 512, 1), (512, 1, 1), 0)
        del buf3
        buf6 = reinterpret_tensor(buf1, (64, 512, 128), (65536, 128, 1), 0)
        del buf1
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_addmm_convert_element_type_erf_mul_permute_rsqrt_sub_var_mean_view_2.run(buf5, buf6, arg0_1, arg3_1, arg4_1, buf2, 32768, 128, stream=raw_stream0)
        del arg0_1
        del arg3_1
        del arg4_1
    return (buf2, buf5, reinterpret_tensor(buf6, (32768, 128), (128, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 256], 'dtype': 'float32', 'stride': [256, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32768, 256], 'dtype': 'bfloat16', 'stride': [256, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [64, 512, 128]}, {'kind': 'shape', 'name': None, 'dims': [32768, 128]}]}
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
