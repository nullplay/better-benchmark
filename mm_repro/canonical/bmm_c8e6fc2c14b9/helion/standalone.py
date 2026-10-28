"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_c8e6fc2c14b9
Shape hash: 97e22389
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([768, 200, 200]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op2(buf0: torch.Tensor, buf1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op2(buf0, buf1, buf2)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), '2b', 'c2bxbsizug4e3tswpizwou33crks3ftiuc4zeeq3q6i56pzqnige.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_permute_unbind_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 58982400}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_permute_unbind_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 9830400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 64 % 200
    x0 = xindex % 64
    x2 = xindex // 12800
    x3 = xindex
    tmp0 = x1.to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x1.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1], 196, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (x0 + 64 * (x2 % 6) + 1152 * x1 + 225792 * (x2 // 6)), tmp6, other=0.0).to(tl.float32)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1], 200, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.full([1], 0.0, tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp6, tmp7, tmp13)
    tl.store(out_ptr0 + x3, tmp14, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'y': 65536, 'x': 256}, tile_hint=TileHint.SQUARE, filename=os.path.join(cache_dir(), 'gz', 'cgzja75hpxkq2sgfrbxoelov5da6djd3zpe7bqbcz4ws5ayofcl4.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_permute_unbind_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 19660800, 'x': 39321600}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_permute_unbind_view_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 49152
    xnumel = 200
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 64
    y1 = yindex // 64
    y3 = yindex
    tmp0 = x2.to(tl.int32)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x2.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1, 1], 196, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (384 + y0 + 64 * (y1 % 6) + 1152 * x2 + 225792 * (y1 // 6)), tmp6 & xmask, other=0.0).to(tl.float32)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1, 1], 200, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.full([1, 1], 0.0, tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp6, tmp7, tmp13)
    tl.store(out_ptr0 + (x2 + 200 * y3), tmp14, xmask)
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

@triton_heuristics.pointwise(size_hints={'x': 33554432}, filename=os.path.join(cache_dir(), 'b5', 'cb5vqvkq262t4p44yvemofypiqdogvjvkgz4abzb2lgtyyj4oo2e.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_abs_convert_element_type_eq_logical_not_mul_ne_slice_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 3, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 590069760}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_abs_convert_element_type_eq_logical_not_mul_ne_slice_view_3(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK: tl.constexpr):
    xnumel = 29503488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 196
    x1 = xindex // 196 % 196
    x2 = xindex // 38416
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 200 * x1 + 40000 * x2), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 0.125, tl.float32)
    tmp3 = tmp0 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4 == tmp4
    tmp6 = tl_math.abs(tmp4)
    tmp7 = tl.full([1], float('inf'), tl.float32)
    tmp8 = tmp6 != tmp7
    tmp9 = tmp5 & tmp8
    tmp10 = tmp9 == 0
    tl.store(out_ptr0 + x3, tmp1, None)
    tl.store(out_ptr1 + x3, tmp4, None)
    tl.store(out_ptr2 + x3, tmp10, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (128, 1152, 14, 14), (225792, 1, 16128, 1152), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((768, 200, 64), (12800, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_constant_pad_nd_permute_unbind_view_0.run(arg0_1, buf0, 9830400, stream=raw_stream0)
        buf1 = empty_strided_cuda((768, 64, 200), (12800, 200, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_constant_pad_nd_permute_unbind_view_1.run(arg0_1, buf1, 49152, 200, stream=raw_stream0)
        del arg0_1
        buf2 = empty_strided_cuda((768, 200, 200), (40000, 200, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2(buf0, buf1, buf2)
        del buf0
        del buf1
        buf3 = empty_strided_cuda((128, 6, 196, 196), (230496, 38416, 196, 1), torch.float32)
        buf4 = empty_strided_cuda((128, 6, 196, 196), (230496, 38416, 196, 1), torch.float32)
        buf5 = empty_strided_cuda((128, 6, 196, 196), (230496, 38416, 196, 1), torch.bool)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_abs_convert_element_type_eq_logical_not_mul_ne_slice_view_3.run(buf2, buf3, buf4, buf5, 29503488, stream=raw_stream0)
    return (buf2, buf4, buf3, buf5)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [128, 1152, 14, 14], 'dtype': 'bfloat16', 'stride': [225792, 1, 16128, 1152], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [128, 3, 6, 64, 196]}, {'kind': 'shape', 'name': None, 'dims': [128, 6, 196, 64]}, {'kind': 'shape', 'name': None, 'dims': [768, 196, 64]}, {'kind': 'shape', 'name': None, 'dims': [128, 6, 64, 196]}, {'kind': 'shape', 'name': None, 'dims': [768, 64, 196]}, {'kind': 'shape', 'name': None, 'dims': [0, 0, 0, 4, 0, 0]}, {'kind': 'shape', 'name': None, 'dims': [0, 4, 0, 0, 0, 0]}, {'kind': 'shape', 'name': None, 'dims': [128, 6, 196, 196]}]}
COMPILED_INPUT_ORDER = [0]

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
