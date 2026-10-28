"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_77d7f41767e0
Shape hash: 08a60a4a
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 2]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(2048):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg1_1: torch.Tensor, arg0_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg1_1, reinterpret_tensor(arg0_1, (2048, 2), (1, 2048), 0), buf0)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 32, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'ht', 'cht7dgl2zhue4tmf5lf35fhglezpe635ac24bi7ckw44klehqhnj.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_argmax_convert_element_type_iota_mul_ne_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 512, 'r0_': 32768}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_argmax_convert_element_type_iota_mul_ne_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 32
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128 * x0), xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tmp2.to(tl.int32)
    tmp4 = r0_1.to(tl.int32)
    tmp5 = tmp4 * tmp3
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, -2147483648)
    tmp9 = tl.broadcast_to(rindex, tmp8.shape)
    tmp7_val, tmp7_idx = triton_helpers.max_with_first_index(tmp8, tmp9, 1)
    tmp7 = tmp7_idx[:, None]
    tl.store(out_ptr0 + x0, tmp7, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 64}, filename=os.path.join(cache_dir(), 'mu', 'cmunq4eu2kd2q5dppcaxcreuun36q4x3msupwtwp2z2tqwlw2f7d.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_index_iota_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 1024}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_index_iota_view_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 2
    x0 = xindex % 2
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x1, xmask, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 128, tl.int32).to(tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 128) | ~xmask, 'index out of bounds: 0 <= tmp4 < 128')
    tmp6 = tl.load(in_ptr1 + (x0 + 2 * tmp4 + 256 * x1), xmask).to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp6, xmask)
    tl.store(out_ptr1 + x2, tmp7, xmask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg1_1, arg0_1), ((4096, 2048), (2, 2048)), ((2048, 1), (2048, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((4096, 2), (2, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg1_1, arg0_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride(arg2_1, (32, 128), (128, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((32,), (1,), torch.int64)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_argmax_convert_element_type_iota_mul_ne_1.run(arg2_1, buf1, 32, 128, stream=raw_stream0)
        del arg2_1
        buf2 = empty_strided_cuda((32, 2), (2, 1), torch.bfloat16)
        buf3 = empty_strided_cuda((32, 2), (2, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_index_iota_view_2.run(buf1, buf0, buf2, buf3, 64, stream=raw_stream0)
        del buf0
        del buf1
    return (buf2, buf3)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [2, 2048], 'dtype': 'bfloat16', 'stride': [2048, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 2048], 'dtype': 'bfloat16', 'stride': [2048, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128], 'dtype': 'int64', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 2]}]}
COMPILED_INPUT_ORDER = [0, 1, 2]

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
