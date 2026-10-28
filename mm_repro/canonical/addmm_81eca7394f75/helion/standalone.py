"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_81eca7394f75
Shape hash: 26e047eb
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([5, 5]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg3_1: torch.Tensor, arg1_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg3_1, reinterpret_tensor(arg1_1, (64, 5), (1, 64), 0), buf0)
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

@triton_heuristics.pointwise(size_hints={'x': 8}, filename=os.path.join(cache_dir(), 'ki', 'cki4mosmcjab3jwcx7xjpasgj2bjsrwr7haw5y7h4n5uexga2h7e.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_convert_element_type_permute_prepare_softmax_online_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 10, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 50}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_convert_element_type_permute_prepare_softmax_online_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + 0).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp4 = tl.load(in_ptr1 + 5 * x0, xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr0 + 1).to(tl.float32)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK])
    tmp11 = tl.load(in_ptr1 + (1 + 5 * x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr0 + 2).to(tl.float32)
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp19 = tl.load(in_ptr1 + (2 + 5 * x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr0 + 3).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK])
    tmp27 = tl.load(in_ptr1 + (3 + 5 * x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp32 = tl.load(in_ptr0 + 4).to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [XBLOCK])
    tmp35 = tl.load(in_ptr1 + (4 + 5 * x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.full([1], 1.0, tl.float32)
    tmp3 = tmp2 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = tmp3 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp10 = tmp2 * tmp9
    tmp12 = tmp2 * tmp11
    tmp13 = tmp10 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tl.maximum(tmp7, tmp14, tl.PropagateNan.ALL)
    tmp18 = tmp2 * tmp17
    tmp20 = tmp2 * tmp19
    tmp21 = tmp18 + tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl.maximum(tmp15, tmp22, tl.PropagateNan.ALL)
    tmp26 = tmp2 * tmp25
    tmp28 = tmp2 * tmp27
    tmp29 = tmp26 + tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tl.maximum(tmp23, tmp30, tl.PropagateNan.ALL)
    tmp34 = tmp2 * tmp33
    tmp36 = tmp2 * tmp35
    tmp37 = tmp34 + tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tl.maximum(tmp31, tmp38, tl.PropagateNan.ALL)
    tmp40 = tmp7 - tmp39
    tmp41 = libdevice.exp(tmp40)
    tmp42 = tmp14 - tmp39
    tmp43 = libdevice.exp(tmp42)
    tmp44 = tmp41 + tmp43
    tmp45 = tmp22 - tmp39
    tmp46 = libdevice.exp(tmp45)
    tmp47 = tmp44 + tmp46
    tmp48 = tmp30 - tmp39
    tmp49 = libdevice.exp(tmp48)
    tmp50 = tmp47 + tmp49
    tmp51 = tmp38 - tmp39
    tmp52 = libdevice.exp(tmp51)
    tmp53 = tmp50 + tmp52
    tl.store(out_ptr0 + x0, tmp39, xmask)
    tl.store(out_ptr1 + x0, tmp53, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 1}, filename=os.path.join(cache_dir(), 'ei', 'ceio5fxcdpw4kdw4ylkcxoajhf3o6ltn5ejyufwb4ndinkc7rch2.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'xnumel': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'xnumel': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_convert_element_type_div_full_gather_log_ne_neg_permute_squeeze_sub_sum_unsqueeze_where_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 15, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_convert_element_type_div_full_gather_log_ne_neg_permute_squeeze_sub_sum_unsqueeze_where_2(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    tmp0 = tl.load(in_ptr0 + 0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp14 = tl.load(in_ptr3 + 0)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp17 = tl.load(in_ptr4 + 0)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK])
    tmp25 = tl.load(in_ptr0 + 1)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK])
    tmp36 = tl.load(in_ptr3 + 1)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK])
    tmp39 = tl.load(in_ptr4 + 1)
    tmp40 = tl.broadcast_to(tmp39, [XBLOCK])
    tmp47 = tl.load(in_ptr0 + 2)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK])
    tmp58 = tl.load(in_ptr3 + 2)
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK])
    tmp61 = tl.load(in_ptr4 + 2)
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK])
    tmp69 = tl.load(in_ptr0 + 3)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK])
    tmp80 = tl.load(in_ptr3 + 3)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK])
    tmp83 = tl.load(in_ptr4 + 3)
    tmp84 = tl.broadcast_to(tmp83, [XBLOCK])
    tmp91 = tl.load(in_ptr0 + 4)
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK])
    tmp102 = tl.load(in_ptr3 + 4)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK])
    tmp105 = tl.load(in_ptr4 + 4)
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK])
    tmp2 = tl.full([1], -100, tl.int64)
    tmp3 = tmp1 != tmp2
    tmp4 = tl.full([1], 0, tl.int64)
    tmp5 = tl.where(tmp3, tmp1, tmp4)
    tl.device_assert((0 <= tmp5) & (tmp5 < 5), 'index out of bounds: 0 <= tmp5 < 5')
    tmp7 = tl.load(in_ptr1 + tmp5, None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp8 * tmp7
    tmp10 = tl.load(in_ptr2 + tmp5, None, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp9 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp16 = tmp13 - tmp15
    tmp19 = tl_math.log(tmp18)
    tmp20 = tmp16 - tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = -tmp21
    tmp23 = tl.full([1], 0.0, tl.float32)
    tmp24 = tl.where(tmp3, tmp22, tmp23)
    tmp27 = tmp26 != tmp2
    tmp28 = tl.where(tmp27, tmp26, tmp4)
    tl.device_assert((0 <= tmp28) & (tmp28 < 5), 'index out of bounds: 0 <= tmp28 < 5')
    tmp30 = tl.load(in_ptr1 + tmp28, None, eviction_policy='evict_last').to(tl.float32)
    tmp31 = tmp8 * tmp30
    tmp32 = tl.load(in_ptr2 + (5 + tmp28), None, eviction_policy='evict_last').to(tl.float32)
    tmp33 = tmp8 * tmp32
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp38 = tmp35 - tmp37
    tmp41 = tl_math.log(tmp40)
    tmp42 = tmp38 - tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp44 = -tmp43
    tmp45 = tl.where(tmp27, tmp44, tmp23)
    tmp46 = tmp24 + tmp45
    tmp49 = tmp48 != tmp2
    tmp50 = tl.where(tmp49, tmp48, tmp4)
    tl.device_assert((0 <= tmp50) & (tmp50 < 5), 'index out of bounds: 0 <= tmp50 < 5')
    tmp52 = tl.load(in_ptr1 + tmp50, None, eviction_policy='evict_last').to(tl.float32)
    tmp53 = tmp8 * tmp52
    tmp54 = tl.load(in_ptr2 + (10 + tmp50), None, eviction_policy='evict_last').to(tl.float32)
    tmp55 = tmp8 * tmp54
    tmp56 = tmp53 + tmp55
    tmp57 = tmp56.to(tl.float32)
    tmp60 = tmp57 - tmp59
    tmp63 = tl_math.log(tmp62)
    tmp64 = tmp60 - tmp63
    tmp65 = tmp64.to(tl.float32)
    tmp66 = -tmp65
    tmp67 = tl.where(tmp49, tmp66, tmp23)
    tmp68 = tmp46 + tmp67
    tmp71 = tmp70 != tmp2
    tmp72 = tl.where(tmp71, tmp70, tmp4)
    tl.device_assert((0 <= tmp72) & (tmp72 < 5), 'index out of bounds: 0 <= tmp72 < 5')
    tmp74 = tl.load(in_ptr1 + tmp72, None, eviction_policy='evict_last').to(tl.float32)
    tmp75 = tmp8 * tmp74
    tmp76 = tl.load(in_ptr2 + (15 + tmp72), None, eviction_policy='evict_last').to(tl.float32)
    tmp77 = tmp8 * tmp76
    tmp78 = tmp75 + tmp77
    tmp79 = tmp78.to(tl.float32)
    tmp82 = tmp79 - tmp81
    tmp85 = tl_math.log(tmp84)
    tmp86 = tmp82 - tmp85
    tmp87 = tmp86.to(tl.float32)
    tmp88 = -tmp87
    tmp89 = tl.where(tmp71, tmp88, tmp23)
    tmp90 = tmp68 + tmp89
    tmp93 = tmp92 != tmp2
    tmp94 = tl.where(tmp93, tmp92, tmp4)
    tl.device_assert((0 <= tmp94) & (tmp94 < 5), 'index out of bounds: 0 <= tmp94 < 5')
    tmp96 = tl.load(in_ptr1 + tmp94, None, eviction_policy='evict_last').to(tl.float32)
    tmp97 = tmp8 * tmp96
    tmp98 = tl.load(in_ptr2 + (20 + tmp94), None, eviction_policy='evict_last').to(tl.float32)
    tmp99 = tmp8 * tmp98
    tmp100 = tmp97 + tmp99
    tmp101 = tmp100.to(tl.float32)
    tmp104 = tmp101 - tmp103
    tmp107 = tl_math.log(tmp106)
    tmp108 = tmp104 - tmp107
    tmp109 = tmp108.to(tl.float32)
    tmp110 = -tmp109
    tmp111 = tl.where(tmp93, tmp110, tmp23)
    tmp112 = tmp90 + tmp111
    tmp113 = tmp3.to(tl.int64)
    tmp114 = tmp27.to(tl.int64)
    tmp115 = tmp113 + tmp114
    tmp116 = tmp49.to(tl.int64)
    tmp117 = tmp115 + tmp116
    tmp118 = tmp71.to(tl.int64)
    tmp119 = tmp117 + tmp118
    tmp120 = tmp93.to(tl.int64)
    tmp121 = tmp119 + tmp120
    tmp122 = tmp121.to(tl.float32)
    tmp123 = tmp112 / tmp122
    tl.store(in_out_ptr0 + tl.full([XBLOCK], 0, tl.int32).broadcast_to(XBLOCK), tmp123, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg3_1, arg1_1), ((5, 64), (5, 64)), ((64, 1), (64, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((5, 5), (5, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg3_1, arg1_1, buf0)
        del arg1_1
        del arg3_1
        assert_size_stride(arg2_1, (5,), (1,), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((5, 1), (1, 5), torch.float32)
        buf2 = empty_strided_cuda((5, 1), (1, 5), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_convert_element_type_permute_prepare_softmax_online_1.run(arg2_1, buf0, buf1, buf2, 5, stream=raw_stream0)
        assert_size_stride(arg0_1, (5,), (1,), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf3 = empty_strided_cuda((), (), torch.bfloat16)
        buf4 = buf3
        del buf3
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_convert_element_type_div_full_gather_log_ne_neg_permute_squeeze_sub_sum_unsqueeze_where_2.run(buf4, arg0_1, arg2_1, buf0, buf1, buf2, 1, stream=raw_stream0)
        del arg0_1
        del arg2_1
        del buf0
        del buf1
        del buf2
    return (buf4,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [5], 'dtype': 'int64', 'stride': [1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 5}, 'generator': {'kind': 'index', 'low': 0, 'high': 5}}, {'kind': 'tensor', 'name': None, 'shape': [5, 64], 'dtype': 'bfloat16', 'stride': [64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [5], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [5, 64], 'dtype': 'bfloat16', 'stride': [64, 1], 'device': 'cuda', 'storage_offset': 0}]}
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
