"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_089926541349
Shape hash: 79b88292
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([32768, 30528]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op2(arg2_1: torch.Tensor, buf0: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op2(arg2_1, reinterpret_tensor(buf0, (512, 30528), (30528, 1), 0), buf2)
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

@triton_heuristics.pointwise(size_hints={'y': 512, 'x': 32768}, tile_hint=TileHint.DEFAULT, filename=os.path.join(cache_dir(), '3y', 'c3yxpk2qwlz4mqomhh2y64i4qzjuke3b2htbhp25jm6afr3qtjsu.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_cat_convert_element_type_permute_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 15627264, 'x': 109390848}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_cat_convert_element_type_permute_0(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 30522
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex
    x1 = xindex
    tmp0 = y0.to(tl.int32)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = y0.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1, 1], 128, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = tl.load(in_ptr0 + (128 * x1 + y0), tmp6 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp0 >= tmp5
    tmp9 = tl.full([1, 1], 512, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + (x1 + 30522 * (-128 + y0)), tmp8 & xmask & ymask, other=0.0)
    tmp12 = tl.where(tmp6, tmp7, tmp11)
    tmp13 = tmp12.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 30528 * y0), tmp13, xmask & ymask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 4096}, filename=os.path.join(cache_dir(), 'xp', 'cxpuv7ff2vkoazj6pkp4insycdenuvwq2tvvitkzktmlsdoexz6z.py'), triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_cat_constant_pad_nd_convert_element_type_permute_1', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 12288}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_cat_constant_pad_nd_convert_element_type_permute_1(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = xindex // 6
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (30522 + x0 + 30528 * x1), tmp0, xmask)
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

@triton_heuristics.pointwise(size_hints={'x': 1073741824}, filename=os.path.join(cache_dir(), 'nl', 'cnlhe2xi2upd4amjpsrk6kcmvg25pb2ii4l7hgvpwjer7xrlzhxl.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_convert_element_type_slice_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 16002440424}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_convert_element_type_slice_view_3(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1000144896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 30522
    x1 = xindex // 30522
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 30528 * x1), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + x0, None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp4, None)
    tl.store(out_ptr1 + x2, tmp5, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((30522, 128), (384, 30522)), ((128, 1), (30522, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((512, 30528), (30528, 1), torch.bfloat16).as_strided((512, 30522), (30528, 1))
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convert_element_type_permute_0.run(arg0_1, arg1_1, buf0, 512, 30522, stream=raw_stream0)
        del arg0_1
        del arg1_1
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_cat_constant_pad_nd_convert_element_type_permute_1.run(buf0, 3072, stream=raw_stream0)
        assert_size_stride(arg2_1, (32768, 512), (512, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf2 = empty_strided_cuda((32768, 30528), (30528, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2(arg2_1, buf0, buf2)
        del arg2_1
        del buf0
        assert_size_stride(arg3_1, (30522,), (1,), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf3 = empty_strided_cuda((256, 128, 30522), (3906816, 30522, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((32768, 30522), (30522, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_convert_element_type_slice_view_3.run(buf2, arg3_1, buf3, buf4, 1000144896, stream=raw_stream0)
        del arg3_1
        del buf2
    return (buf3, buf4)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [30522, 128], 'dtype': 'float32', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [384, 30522], 'dtype': 'float32', 'stride': [30522, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32768, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [30522], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [0, 6, 0, 0]}, {'kind': 'shape', 'name': None, 'dims': [256, 128, 30522]}, {'kind': 'shape', 'name': None, 'dims': [32768, 30522]}, {'kind': 'shape', 'name': None, 'dims': [256, 128, 30522]}]}
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
