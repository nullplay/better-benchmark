"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_5cd3c18b3f94
Shape hash: 8afdfb1f
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([25088, 512]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1536):
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

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), 'sm', 'csmopxnbecoxwbs3vg5qpma5jxqqom7mbhdsob3et246qxbp5fgy.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_convert_element_type_mul_permute_sub_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 4, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 488515584}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_convert_element_type_mul_permute_sub_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK: tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 512
    x1 = xindex // 512 % 14
    x2 = xindex // 7168 % 14
    x3 = xindex // 100352
    x4 = xindex
    x5 = xindex // 512
    tmp0 = tl.load(in_ptr0 + (x0 + 512 * (x1 % 7) + 3584 * (x2 % 7) + 25088 * (x1 // 7) + 50176 * (x2 // 7) + 100352 * x3), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + x4, None).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + x5, None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + x5, None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = tl.full([1], 512.0, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 - tmp8
    tmp11 = tmp9 * tmp10
    tmp12 = tmp3 * tmp11
    tmp13 = tmp1 * tmp11
    tl.store(out_ptr0 + x4, tmp3, None)
    tl.store(out_ptr1 + x4, tmp5, None)
    tl.store(out_ptr2 + x4, tmp12, None)
    tl.store(out_ptr3 + x4, tmp13, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((25088, 1536), (1536, 512)), ((1536, 1), (512, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((25088, 512), (512, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1, arg4_1, arg5_1), ((512,), (128, 14, 14, 512), (128, 14, 14, 1), (128, 14, 14, 1)), ((1,), (100352, 7168, 512, 1), (196, 14, 1, 1), (196, 14, 1, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        buf1 = empty_strided_cuda((128, 14, 14, 512), (100352, 7168, 512, 1), torch.float32)
        buf2 = empty_strided_cuda((128, 14, 14, 512), (100352, 7168, 512, 1), torch.float32)
        buf3 = empty_strided_cuda((128, 14, 14, 512), (100352, 7168, 512, 1), torch.float32)
        buf4 = empty_strided_cuda((128, 14, 14, 512), (100352, 7168, 512, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convert_element_type_mul_permute_sub_view_1.run(buf0, arg2_1, arg3_1, arg4_1, arg5_1, buf1, buf2, buf3, buf4, 12845056, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf0
    return (buf1, buf2, buf3, buf4)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [25088, 1536], 'dtype': 'bfloat16', 'stride': [1536, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1536, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 14, 14, 512], 'dtype': 'bfloat16', 'stride': [100352, 7168, 512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 14, 14, 1], 'dtype': 'float32', 'stride': [196, 14, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 14, 14, 1], 'dtype': 'float32', 'stride': [196, 14, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [512, 49, 512]}, {'kind': 'shape', 'name': None, 'dims': [512, 7, 7, 512]}, {'kind': 'shape', 'name': None, 'dims': [128, 2, 2, 7, 7, 512]}, {'kind': 'shape', 'name': None, 'dims': [128, 14, 14, 512]}]}
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
