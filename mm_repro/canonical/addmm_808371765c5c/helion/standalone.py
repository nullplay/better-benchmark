"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_808371765c5c
Shape hash: 1bbea10a
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([131072, 432]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(144):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg2_1: torch.Tensor, arg0_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg2_1, reinterpret_tensor(arg0_1, (144, 432), (1, 144), 0), buf0)
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
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 33554432}, filename=os.path.join(cache_dir(), 'vj', 'cvje6g6i7pb2dvoueozr7arcssbgpwspxueyvvzytig45765dzwz.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 113246496}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 36
    x2 = xindex // 9216 % 4
    x1 = xindex // 36 % 256
    x3 = xindex // 36864
    x4 = xindex // 36
    tmp0 = tl.load(in_ptr0 + (x0 + 36 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0 + 36 * x2 + 432 * x1 + 110592 * x3), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tl.store(out_ptr0 + (x0 + 40 * x4), tmp5, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 2097152}, filename=os.path.join(cache_dir(), 'a3', 'ca3rq5igpicbfrfi4e3h4pttrqnom47fkikzhsikefnuxjw4qs3b.py'), triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 8388608}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_2(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 4
    x1 = xindex // 4
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + (x0 + 40 * x1), tmp0, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 33554432}, filename=os.path.join(cache_dir(), 'ci', 'cciw4yulkpv64awhgicpanp3leb2axbdsuesp2jepbe6llolzeqz.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 113246496}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 36
    x2 = xindex // 9216 % 4
    x1 = xindex // 36 % 256
    x3 = xindex // 36864
    x4 = xindex // 36
    tmp0 = tl.load(in_ptr0 + (144 + x0 + 36 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (144 + x0 + 36 * x2 + 432 * x1 + 110592 * x3), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tl.store(out_ptr0 + (x0 + 40 * x4), tmp5, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 33554432}, filename=os.path.join(cache_dir(), '3u', 'c3uzl7cowy3pljzq7pgps7s5apr53ju3ganyr4c7epsemp4w6kos.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 113246496}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 18874368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 36
    x2 = xindex // 9216 % 4
    x1 = xindex // 36 % 256
    x3 = xindex // 36864
    x4 = xindex // 36
    tmp0 = tl.load(in_ptr0 + (288 + x0 + 36 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (288 + x0 + 36 * x2 + 432 * x1 + 110592 * x3), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tl.store(out_ptr0 + (x0 + 40 * x4), tmp5, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg2_1, arg0_1), ((131072, 144), (432, 144)), ((144, 1), (144, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((131072, 432), (432, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg2_1, arg0_1, buf0)
        del arg0_1
        del arg2_1
        assert_size_stride(arg1_1, (432,), (1,), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf3 = empty_strided_cuda((512, 4, 256, 40), (40960, 10240, 40, 1), torch.bfloat16)
        buf1 = reinterpret_tensor(buf3, (512, 4, 256, 36), (40960, 10240, 40, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_1.run(arg1_1, buf0, buf1, 18874368, stream=raw_stream0)
        buf2 = reinterpret_tensor(buf3, (512, 4, 256, 4), (40960, 10240, 40, 1), 36)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_2.run(buf2, 2097152, stream=raw_stream0)
        buf6 = empty_strided_cuda((512, 4, 256, 40), (40960, 10240, 40, 1), torch.bfloat16)
        buf4 = reinterpret_tensor(buf6, (512, 4, 256, 36), (40960, 10240, 40, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_3.run(arg1_1, buf0, buf4, 18874368, stream=raw_stream0)
        buf5 = reinterpret_tensor(buf6, (512, 4, 256, 4), (40960, 10240, 40, 1), 36)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_2.run(buf5, 2097152, stream=raw_stream0)
        buf9 = empty_strided_cuda((512, 4, 256, 40), (40960, 10240, 40, 1), torch.bfloat16)
        buf7 = reinterpret_tensor(buf9, (512, 4, 256, 36), (40960, 10240, 40, 1), 0)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_4.run(arg1_1, buf0, buf7, 18874368, stream=raw_stream0)
        del arg1_1
        del buf0
        buf8 = reinterpret_tensor(buf9, (512, 4, 256, 4), (40960, 10240, 40, 1), 36)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_constant_pad_nd_permute_unbind_view_2.run(buf8, 2097152, stream=raw_stream0)
    return (buf3, buf6, buf9)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [432, 144], 'dtype': 'bfloat16', 'stride': [144, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [432], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [131072, 144], 'dtype': 'bfloat16', 'stride': [144, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [512, 256, 432]}, {'kind': 'shape', 'name': None, 'dims': [512, 256, 3, 4, 36]}, {'kind': 'shape', 'name': None, 'dims': [0, 4]}, {'kind': 'shape', 'name': None, 'dims': [0, 4]}, {'kind': 'shape', 'name': None, 'dims': [0, 4]}]}
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
