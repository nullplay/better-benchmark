"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_187072f1a97b
Shape hash: 06144af8
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op4(A: torch.Tensor, B: torch.Tensor, B1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 1536]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1536):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = B1[tile_r, tile_x].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp1, tmp5)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)
        out1[tile_y, tile_x] = acc1.to(torch.bfloat16)

def call_helion_op1_op4(arg2_1: torch.Tensor, buf0: torch.Tensor, buf3: torch.Tensor, buf1: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op1_op4(arg2_1, reinterpret_tensor(buf0, (1536, 1536), (1, 1536), 0), reinterpret_tensor(buf3, (1536, 1536), (1, 1536), 0), buf1, buf4)
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

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), 'il', 'cilr4vgwfdstsbd53od4qcqy2t2gohnfvxpvm4k2xi24qvg367if.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 18874368}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, None)
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

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'zr', 'czr5jmnrm4mrgiionwomjespxoapsex7ljt2x5gyzouzsktvhjem.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_clone_convert_element_type_permute_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 37754880}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_clone_convert_element_type_permute_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 64
    x2 = xindex // 32768 % 24
    x1 = xindex // 64 % 512
    x3 = xindex // 786432
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + 64 * x2 + 1536 * x1 + 786432 * x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 1.0, tl.float32)
    tmp3 = tmp2 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = tmp3 + tmp5
    tl.store(out_ptr0 + x4, tmp6, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg4_1, (1536, 1536), (1536, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg4_1 = copy_if_misaligned(arg4_1)
        buf0 = empty_strided_cuda((1536, 1536), (1536, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg4_1, buf0, 2359296, stream=raw_stream0)
        del arg4_1
        assert_size_stride(arg1_1, (1536, 1536), (1536, 1), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf3 = empty_strided_cuda((1536, 1536), (1536, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg1_1, buf3, 2359296, stream=raw_stream0)
        del arg1_1
        assert_size_stride(arg2_1, (4096, 1536), (1536, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((4096, 1536), (1536, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((4096, 1536), (1536, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op4(arg2_1, buf0, buf3, buf1, buf4)
        del arg2_1
        del buf0
        del buf3
        assert_size_stride(arg3_1, (1536,), (1,), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf2 = empty_strided_cuda((8, 24, 512, 64), (786432, 32768, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_convert_element_type_permute_view_2.run(arg3_1, buf1, buf2, 6291456, stream=raw_stream0)
        del arg3_1
        assert_size_stride(arg0_1, (1536,), (1,), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf5 = reinterpret_tensor(buf1, (8, 24, 512, 64), (786432, 32768, 64, 1), 0)
        del buf1
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_convert_element_type_permute_view_2.run(arg0_1, buf4, buf5, 6291456, stream=raw_stream0)
        del arg0_1
        del buf4
    return (reinterpret_tensor(buf5, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf2, (192, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf2, (192, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf5, (192, 64, 512), (32768, 1, 64), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1536], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1536, 1536], 'dtype': 'float32', 'stride': [1536, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 1536], 'dtype': 'bfloat16', 'stride': [1536, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1536], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1536, 1536], 'dtype': 'float32', 'stride': [1536, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 1536]}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 24, -1]}, {'kind': 'shape', 'name': None, 'dims': [-1, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 1536]}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 24, -1]}, {'kind': 'shape', 'name': None, 'dims': [-1, 512, 64]}]}
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
