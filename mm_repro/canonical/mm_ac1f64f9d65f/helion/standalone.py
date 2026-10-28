"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_ac1f64f9d65f
Shape hash: 5dd1cecc
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op3_op4(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([2, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(8192):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp4.to(torch.float32)

def call_helion_op3_op4(buf1: torch.Tensor, arg8_1: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op3_op4(reinterpret_tensor(buf1, (2, 8192), (1, 2), 0), arg8_1, buf4)
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

@triton_heuristics.pointwise(size_hints={'x': 16384}, filename=os.path.join(cache_dir(), 'vp', 'cvp37jm6b6nvwqmtfyectusk7zv7dznj3k7iancg66f7oaf2wl66.py'), triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_full_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 65536}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_full_0(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + x0, tmp0, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 16}, filename=os.path.join(cache_dir(), '44', 'c44vsfelrdrj5abu45egxucezgryuziptrpsl4wvb64f4bnu7bsh.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i1', 'in_ptr2': '*i1', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*i64', 'in_ptr7': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_convert_element_type_div_exp_full_index_put_log_mul_prepare_softmax_online_scalar_tensor_sub_sum_where_1', 'mutated_arg_names': ['out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 12, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 216}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_convert_element_type_div_exp_full_index_put_log_mul_prepare_softmax_online_scalar_tensor_sub_sum_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 2
    x0 = xindex % 2
    tmp0 = tl.load(in_ptr0 + x2, xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + 2 * x1, xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr0 + (1 + 2 * x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr1 + 2 * x1, xmask, eviction_policy='evict_last').to(tl.int1)
    tmp22 = tl.load(in_ptr2 + x1, xmask, eviction_policy='evict_last').to(tl.int1)
    tmp23 = tl.load(in_ptr3 + 0)
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK])
    tmp25 = tl.load(in_ptr4 + 0)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp32 = tl.load(in_ptr1 + (1 + 2 * x1), xmask, eviction_policy='evict_last').to(tl.int1)
    tmp39 = tl.load(in_ptr5 + x1, xmask, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr6 + x1, xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr7 + x2, xmask).to(tl.float32)
    tmp52 = tl.load(in_ptr1 + x2, xmask).to(tl.int1)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tl.maximum(tmp3, tmp5, tl.PropagateNan.ALL)
    tmp7 = tmp1 - tmp6
    tmp8 = tmp3 - tmp6
    tmp9 = libdevice.exp(tmp8)
    tmp10 = tmp5 - tmp6
    tmp11 = libdevice.exp(tmp10)
    tmp12 = tmp9 + tmp11
    tmp13 = tl_math.log(tmp12)
    tmp14 = tmp7 - tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = libdevice.exp(tmp16)
    tmp19 = tl.full([1], -1.0, tl.float32)
    tmp20 = tl.full([1], 0.0, tl.float32)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp27 = tmp24 / tmp26
    tmp28 = tl.where(tmp22, tmp27, tmp20)
    tmp29 = tmp21 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp30.to(tl.float32)
    tmp33 = tl.where(tmp32, tmp19, tmp20)
    tmp34 = tmp33 * tmp28
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp31 + tmp36
    tmp38 = tmp17 * tmp37
    tmp40 = tl.full([XBLOCK], 8, tl.int32).to(tl.int32)
    tmp41 = tmp39 + tmp40
    tmp42 = tmp39 < 0
    tmp43 = tl.where(tmp42, tmp41, tmp39)
    tl.device_assert((0 <= tmp43) & (tmp43 < 8) | ~xmask, 'index out of bounds: 0 <= tmp43 < 8')
    tmp46 = tl.full([XBLOCK], 1024, tl.int32).to(tl.int32)
    tmp47 = tmp45 + tmp46
    tmp48 = tmp45 < 0
    tmp49 = tl.where(tmp48, tmp47, tmp45)
    tl.device_assert((0 <= tmp49) & (tmp49 < 1024) | ~xmask, 'index out of bounds: 0 <= tmp49 < 1024')
    tmp53 = tl.where(tmp52, tmp19, tmp20)
    tmp54 = tmp53 * tmp28
    tmp55 = tmp54.to(tl.float32)
    tmp56 = tmp55.to(tl.float32)
    tmp57 = tmp56 - tmp38
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp51 + tmp58
    tl.atomic_add(out_ptr1 + (x0 + 2 * tmp49 + 2048 * tmp43), tmp59, xmask, sem='relaxed')
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
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf1 = empty_strided_cuda((8, 1024, 2), (2048, 2, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_full_0.run(buf1, 16384, stream=raw_stream0)
        assert_size_stride_grouped((arg4_1, arg2_1, arg3_1, arg0_1, arg1_1, arg6_1, arg7_1, arg5_1), ((8, 2), (8, 2), (8, 1), (), (), (8,), (8,), (8, 2)), ((2, 1), (2, 1), (1, 1), (), (), (1,), (1,), (2, 1)), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        arg7_1 = copy_if_misaligned(arg7_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_convert_element_type_div_exp_full_index_put_log_mul_prepare_softmax_online_scalar_tensor_sub_sum_where_1.run(arg4_1, arg2_1, arg3_1, arg0_1, arg1_1, arg6_1, arg7_1, arg5_1, buf1, 16, stream=raw_stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        assert_size_stride(arg8_1, (8192, 768), (768, 1), 'input')
        arg8_1 = copy_if_misaligned(arg8_1)
        buf4 = empty_strided_cuda((2, 768), (768, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op3_op4(buf1, arg8_1, buf4)
        del arg8_1
        del buf1
    return (buf4,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [], 'dtype': 'float32', 'stride': [], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [], 'dtype': 'float32', 'stride': [], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 2], 'dtype': 'bool', 'stride': [2, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 1], 'dtype': 'bool', 'stride': [1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 2], 'dtype': 'bfloat16', 'stride': [2, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 2], 'dtype': 'bfloat16', 'stride': [2, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 8}, 'generator': {'kind': 'index', 'low': 0, 'high': 8}}, {'kind': 'tensor', 'name': None, 'shape': [8], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 1024}, 'generator': {'kind': 'index', 'low': 0, 'high': 1024}}, {'kind': 'tensor', 'name': None, 'shape': [8192, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8, 1024, 2]}, {'kind': 'shape', 'name': None, 'dims': [8192, 2]}]}
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
