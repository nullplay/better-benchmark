"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_a63aa6fb3e22
Shape hash: 6a47ea79
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(buf0: torch.Tensor, arg1_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(reinterpret_tensor(buf0, (8192, 768), (768, 1), 0), reinterpret_tensor(arg1_1, (768, 768), (1, 768), 0), buf1)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op5(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op5(buf4: torch.Tensor, arg3_1: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op5(reinterpret_tensor(buf4, (8192, 768), (768, 1), 0), reinterpret_tensor(arg3_1, (768, 768), (1, 768), 0), buf5)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), '4w', 'c4wnot3n5zzbayg3algxr7za2xaelbc4frcq6mdz5d5crlqn3q7e.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_permute_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 62914560}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_permute_0(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 768
    x1 = xindex // 768 % 8
    x2 = xindex // 6144
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 768 * x2 + 786432 * x1), None).to(tl.float32)
    tl.store(out_ptr0 + x3, tmp0, None)
    tl.store(out_ptr1 + x3, tmp0, None)
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

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'py', 'cpyvamctag6hz3xhdmf5ibr3pzm3qz6djzz4rvimqjsskpavz7cf.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_div_permute_view_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 37750272}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_div_permute_view_2(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = xindex % 64
    x1 = xindex // 64 % 96
    tmp0 = tl.load(in_out_ptr0 + x3, None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + 64 * (x1 % 12)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.125, tl.float32)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + x3, tmp4, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), 'p5', 'cp5benbnrnfu3v7wjjppqjvlkkpxsfac2saujxwsvdqkqsd6qoe5.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_as_strided_clone_unsqueeze_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 50331648}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_as_strided_clone_unsqueeze_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 9437184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 64
    x1 = xindex // 64 % 512
    x2 = xindex // 32768 % 3
    x3 = xindex // 98304
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x3 + 6144 * x1 + 1572864 * x2), None).to(tl.float32)
    tl.store(out_ptr0 + x4, tmp0, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), '7f', 'c7f3hmawg7k7gaq6u7tyirs4pub3qqnpftn56kx3ibgezb3p6woc.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_permute_view_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 37750272}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_permute_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = xindex % 64
    x1 = xindex // 64 % 96
    tmp0 = tl.load(in_out_ptr0 + x3, None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0 + 64 * (x1 % 12)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + x3, tmp2, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'y': 32768, 'x': 512}, tile_hint=TileHint.SQUARE, filename=os.path.join(cache_dir(), 'b4', 'cb44oru7xshb7vbnk7l6d4dfe7mkt2otdgcp3skya3el6jtcivet.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_as_strided_clone_permute_unsqueeze_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 12582912, 'x': 37748736}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_as_strided_clone_permute_unsqueeze_5(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 18432
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 64
    y1 = yindex // 64 % 3
    y2 = yindex // 192
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64 * y2 + 6144 * x3 + 1572864 * y1), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x3 + 512 * y4), tmp0, xmask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 1024, 768), (786432, 768, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((1024, 8, 768), (6144, 768, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((1024, 8, 768), (6144, 768, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_0.run(arg0_1, buf0, buf4, 6291456, stream=raw_stream0)
        del arg0_1
        assert_size_stride(arg1_1, (768, 768), (768, 1), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((8192, 768), (768, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(buf0, arg1_1, buf1)
        del arg1_1
        del buf0
        assert_size_stride(arg2_1, (768,), (1,), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf2 = reinterpret_tensor(buf1, (96, 2, 512, 64), (64, 3145728, 6144, 1), 0)
        del buf1
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_permute_view_2.run(buf2, arg2_1, 6291456, stream=raw_stream0)
        del arg2_1
        buf3 = empty_strided_cuda((96, 3, 512, 64, 1), (98304, 32768, 64, 1, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_as_strided_clone_unsqueeze_3.run(buf2, buf3, 9437184, stream=raw_stream0)
        assert_size_stride(arg3_1, (768, 768), (768, 1), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf5 = reinterpret_tensor(buf2, (8192, 768), (768, 1), 0)
        del buf2
        raw_stream0 = get_raw_stream(0)
        call_helion_op5(buf4, arg3_1, buf5)
        del arg3_1
        del buf4
        assert_size_stride(arg4_1, (768,), (1,), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        buf6 = reinterpret_tensor(buf5, (96, 2, 512, 64), (64, 3145728, 6144, 1), 0)
        del buf5
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_permute_view_4.run(buf6, arg4_1, 6291456, stream=raw_stream0)
        del arg4_1
        buf7 = empty_strided_cuda((96, 3, 64, 512, 1), (98304, 32768, 512, 1, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_as_strided_clone_permute_unsqueeze_5.run(buf6, buf7, 18432, 512, stream=raw_stream0)
        del buf6
    return (reinterpret_tensor(buf3, (288, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf7, (288, 64, 512), (32768, 512, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [8, 1024, 768], 'dtype': 'bfloat16', 'stride': [786432, 768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8192, 768]}, {'kind': 'shape', 'name': None, 'dims': [1024, 8, 768]}, {'kind': 'shape', 'name': None, 'dims': [1024, 8, 12, 64]}, {'kind': 'shape', 'name': None, 'dims': [96, 1024, 64]}, {'kind': 'shape', 'name': None, 'dims': [96, 2, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [96, 3, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [64, 1572864, 6144, 1]}, {'kind': 'shape', 'name': None, 'dims': [288, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [8192, 768]}, {'kind': 'shape', 'name': None, 'dims': [1024, 8, 768]}, {'kind': 'shape', 'name': None, 'dims': [1024, 8, 12, 64]}, {'kind': 'shape', 'name': None, 'dims': [96, 1024, 64]}, {'kind': 'shape', 'name': None, 'dims': [96, 2, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [96, 3, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [64, 1572864, 6144, 1]}, {'kind': 'shape', 'name': None, 'dims': [288, 64, 512]}]}
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
