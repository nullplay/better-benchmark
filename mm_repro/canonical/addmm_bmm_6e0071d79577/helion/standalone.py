"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_bmm_6e0071d79577
Shape hash: b63cf89e
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([32768, 56, 32]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(56):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg0_1, arg1_1, buf0)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([401408, 128]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op2(buf1: torch.Tensor, arg2_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op2(reinterpret_tensor(buf1, (401408, 128), (128, 1), 0), reinterpret_tensor(arg2_1, (128, 128), (1, 128), 0), buf2)
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

@triton_heuristics.pointwise(size_hints={'x': 67108864}, filename=os.path.join(cache_dir(), 'zv', 'czvjz4n7np6grl45mciyakh3iw2d2celxg5vbgod5qchnelpe3vh.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_permute_slice_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 308281344}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_permute_slice_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 32
    x1 = xindex // 32 % 4
    x2 = xindex // 128 % 49
    x3 = xindex // 6272
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32 * x2 + 1792 * x1 + 7168 * x3), None).to(tl.float32)
    tl.store(out_ptr0 + x4, tmp0, None)
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

@triton_heuristics.pointwise(size_hints={'x': 67108864}, filename=os.path.join(cache_dir(), 'tf', 'ctf24ext5lidnpeq2dv5jp2ptyfcaidgork6tkfcypomyo2fybky.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_clone_fmod_index_iota_permute_slice_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 308281600}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_clone_fmod_index_iota_permute_slice_view_3(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 51380224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 128
    x1 = xindex // 128 % 56
    x2 = xindex // 7168 % 56
    x3 = xindex // 401408
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0 + 128 * ((53 + x1) % 7) + 896 * ((53 + x2) % 7) + 6272 * ((53 + x1) // 7 % 8) + 50176 * ((53 + x2) // 7 % 8) + 401408 * x3), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tl.store(out_ptr0 + x4, tmp5, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((32768, 56, 56), (32768, 56, 32)), ((3136, 56, 1), (1792, 32, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((32768, 56, 32), (1792, 32, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((8192, 49, 4, 32), (6272, 128, 32, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_slice_view_1.run(buf0, buf1, 51380224, stream=raw_stream0)
        del buf0
        assert_size_stride(arg2_1, (128, 128), (128, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf2 = empty_strided_cuda((401408, 128), (128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2(buf1, arg2_1, buf2)
        del arg2_1
        assert_size_stride(arg3_1, (128,), (1,), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf3 = reinterpret_tensor(buf1, (128, 56, 56, 128), (401408, 7168, 128, 1), 0)
        del buf1
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_clone_fmod_index_iota_permute_slice_view_3.run(arg3_1, buf2, buf3, 51380224, stream=raw_stream0)
        del arg3_1
        del buf2
    return (buf3,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [32768, 56, 56], 'dtype': 'bfloat16', 'stride': [3136, 56, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32768, 56, 32], 'dtype': 'bfloat16', 'stride': [1792, 32, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 128], 'dtype': 'bfloat16', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8192, 4, 49, 32]}, {'kind': 'shape', 'name': None, 'dims': [8192, 49, 128]}, {'kind': 'shape', 'name': None, 'dims': [401408, 128]}, {'kind': 'shape', 'name': None, 'dims': [8192, 49, 128]}, {'kind': 'shape', 'name': None, 'dims': [-1, 7, 7, 128]}, {'kind': 'shape', 'name': None, 'dims': [-1, 8, 8, 7, 7, 128]}, {'kind': 'shape', 'name': None, 'dims': [-1, 56, 56, 128]}]}
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
