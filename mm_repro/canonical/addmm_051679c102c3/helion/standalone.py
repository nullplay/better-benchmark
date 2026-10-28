"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_051679c102c3
Shape hash: 023c1956
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op4_op5(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 50272]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = tile_x.index.to(torch.int64)
        tmp5 = hl.full([], 0, dtype=torch.int64)
        tmp6 = tmp4 >= tmp5
        tmp7 = hl.full([], 50265, dtype=torch.int64)
        tmp8 = tmp4 < tmp7
        tmp9 = hl.load(in0, [tile_x.index], extra_mask=tmp8).to(torch.float32)
        tmp10 = tmp9.to(torch.float32)
        tmp11 = torch.full_like(tmp10, 0.0)
        tmp12 = torch.where(tmp8, tmp10, tmp11)
        tmp13 = tmp4 >= tmp7
        tmp14 = hl.full([], 50272, dtype=torch.int64)
        tmp15 = tmp4 < tmp14
        tmp16 = hl.full([], 0.0, dtype=torch.float32)
        tmp17 = torch.full_like(tmp16, 0.0)
        tmp18 = torch.where(tmp13, tmp16, tmp17)
        tmp19 = torch.where(tmp8, tmp12, tmp18)
        tmp20 = hl.full([], 1.0, dtype=torch.float32)
        tmp21 = tmp20 * tmp19
        tmp22 = tmp20 * acc
        tmp23 = tmp21.reshape([tile_x, 1]) + tmp22.permute(1, 0)
        out0[tile_y, tile_x] = tmp23.permute(1, 0).to(torch.bfloat16)

def call_helion_op4_op5(arg2_1: torch.Tensor, buf3: torch.Tensor, arg0_1: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op4_op5(arg2_1, buf3, arg0_1, buf5)
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

@triton_heuristics.pointwise(size_hints={'y': 1024, 'x': 65536}, tile_hint=TileHint.SQUARE, filename=os.path.join(cache_dir(), 'kn', 'cknoc2nxsj7yyvjrarjsdfvfrj5zfkltupk6jejv4lkk4kmbqp7a.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr1': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_constant_pad_nd_convert_element_type_permute_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 154414080, 'x': 154414080}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_constant_pad_nd_convert_element_type_permute_0(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 768
    xnumel = 50265
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 768 * x1), xmask & ymask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr1 + (x1 + 50304 * y0), tmp1, xmask & ymask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 8192}, filename=os.path.join(cache_dir(), '67', 'c67ey5pea67vrieshjkjv7nfv62rbltbnqyko5ysmip5k3athx2x.py'), triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_constant_pad_nd_convert_element_type_permute_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 21504}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_constant_pad_nd_convert_element_type_permute_1(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = xindex // 7
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0 + 50304 * x1), tmp0, xmask)
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

@triton_heuristics.pointwise(size_hints={'x': 1073741824}, filename=os.path.join(cache_dir(), 'uz', 'cuzmzbt5o544pwwyn2xgdfscrl7lma3ra2fgyko6ms54pupnlknm.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_cat_convert_element_type_full_slice_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 8235417600}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_cat_convert_element_type_full_slice_view_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 823541760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 50265
    x1 = xindex // 50265
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 50272 * x1), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x2, tmp1, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg1_1, (50265, 768), (768, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf3 = empty_strided_cuda((768, 50272), (50304, 1), torch.bfloat16)
        buf1 = reinterpret_tensor(buf3, (768, 50265), (50304, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_convert_element_type_permute_0.run(arg1_1, buf1, 768, 50265, stream=raw_stream0)
        del arg1_1
        buf2 = reinterpret_tensor(buf3, (768, 7), (50304, 1), 50265)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_convert_element_type_permute_1.run(buf2, 5376, stream=raw_stream0)
        assert_size_stride_grouped((arg2_1, arg0_1), ((16384, 768), (50265,)), ((768, 1), (1,)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf5 = empty_strided_cuda((16384, 50272), (50272, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op4_op5(arg2_1, buf3, arg0_1, buf5)
        del arg0_1
        del arg2_1
        del buf1
        del buf2
        del buf3
        buf6 = empty_strided_cuda((32, 512, 50265), (25735680, 50265, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_cat_convert_element_type_full_slice_view_3.run(buf5, buf6, 823541760, stream=raw_stream0)
    return (reinterpret_tensor(buf5, (32, 512, 50265), (25739264, 50272, 1), 0), reinterpret_tensor(buf6, (16384, 50265), (50265, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [50265], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [50265, 768], 'dtype': 'float32', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16384, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [0, 7, 0, 0]}, {'kind': 'shape', 'name': None, 'dims': [7]}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 50265]}, {'kind': 'shape', 'name': None, 'dims': [-1, 50265]}]}
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
