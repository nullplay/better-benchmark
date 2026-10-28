"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_08deccedc20e
Shape hash: 59f10daa
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([32, 1280]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1000):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg0_1, arg1_1, buf0)
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
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 2097152}, filename=os.path.join(cache_dir(), 'pp', 'cpp5a7w43fxesmjek6jupwk43uqmcul25zggf6xtsgxdhzlqlsh5.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*i1', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_convert_element_type_div_expand_full_le_mul_relu_squeeze_sub_unsqueeze_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 36270080}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_convert_element_type_div_expand_full_le_mul_relu_squeeze_sub_unsqueeze_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    x0 = xindex % 1280
    x2 = xindex // 62720
    tmp0 = tl.load(in_ptr0 + x4, None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + x0, None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + x0, None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr4 + x0, None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x0 + 1280 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr6 + (x0 + 1280 * x2), None, eviction_policy='evict_last').to(tl.int1)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 - tmp2
    tmp5 = tmp3 * tmp4
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tl.full([1], 0, tl.int32)
    tmp12 = tl.maximum(tmp11, tmp10, tl.PropagateNan.ALL)
    tmp13 = tl.full([1], 0.0, tl.float32)
    tmp14 = tmp12 <= tmp13
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full([1], 1.25, tl.float32)
    tmp19 = tmp17 * tmp18
    tmp20 = tmp15 * tmp19
    tmp21 = tl.full([1], 0.02040816326530612, tl.float32)
    tmp22 = tmp20 * tmp21
    tmp23 = tl.where(tmp14, tmp13, tmp22)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 * tmp3
    tl.store(out_ptr0 + x4, tmp24, None)
    tl.store(out_ptr1 + x4, tmp25, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((32, 1000), (1000, 1280)), ((1000, 1), (1280, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((32, 1280), (1280, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg2_1), ((32, 1280, 7, 7), (1, 1280, 1, 1), (1, 1280, 1, 1), (1280,), (1280,), (32, 1280)), ((62720, 1, 8960, 1280), (1280, 1, 1, 1), (1280, 1, 1, 1), (1,), (1,), (1280, 1)), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        arg7_1 = copy_if_misaligned(arg7_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((32, 1280, 7, 7), (62720, 1, 8960, 1280), torch.float32)
        buf2 = empty_strided_cuda((32, 1280, 7, 7), (62720, 1, 8960, 1280), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_convert_element_type_div_expand_full_le_mul_relu_squeeze_sub_unsqueeze_where_1.run(arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, buf0, arg2_1, buf1, buf2, 2007040, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf0
    return (buf1, buf2)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [32, 1000], 'dtype': 'bfloat16', 'stride': [1000, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000, 1280], 'dtype': 'bfloat16', 'stride': [1280, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 1280], 'dtype': 'bool', 'stride': [1280, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 1280, 7, 7], 'dtype': 'bfloat16', 'stride': [62720, 1, 8960, 1280], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 1280, 1, 1], 'dtype': 'float32', 'stride': [1280, 1, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 1280, 1, 1], 'dtype': 'float32', 'stride': [1280, 1, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1280], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1280], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 1280, 7, 7]}]}
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
