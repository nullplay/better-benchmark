"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_2f09e235f385
Shape hash: c46d5210
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op4_op5_op6(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = in0[tile_x.index // 64, tile_x.index % 64].to(torch.float32)
        tmp6 = tmp4 + tmp5.reshape([1, tile_x])
        tmp7 = tmp6.to(torch.float32)
        out0[tile_y.index.reshape([tile_y, 1]) // 16, tile_y.index.reshape([tile_y, 1]) % 16, tile_x.index.reshape([1, tile_x]) // 64, tile_x.index.reshape([1, tile_x]) % 64] = tmp7.to(torch.bfloat16)
        tmp8 = acc.to(torch.float32)
        tmp9 = in1[tile_x.index // 64, tile_x.index % 64].to(torch.float32)
        tmp10 = tmp8 + tmp9.reshape([1, tile_x])
        tmp11 = tmp10.to(torch.float32)
        out1[tile_y.index.reshape([tile_y, 1]) // 16, tile_y.index.reshape([tile_y, 1]) % 16, tile_x.index.reshape([1, tile_x]) // 64, tile_x.index.reshape([1, tile_x]) % 64] = tmp11.to(torch.bfloat16)

def call_helion_op4_op5_op6(buf2: torch.Tensor, buf3: torch.Tensor, arg4_1: torch.Tensor, arg3_1: torch.Tensor, buf5: torch.Tensor, buf6: torch.Tensor) -> None:
    helion_op4_op5_op6(reinterpret_tensor(buf2, (8192, 1024), (1024, 1), 0), reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0), arg4_1, arg3_1, buf5, buf6)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
aten = torch.ops.aten
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

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'ij', 'cije3ha5dl2sokmz55kzim6f4c2dt3x4b4iphw6v4rnznlalp6zg.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'out_ptr1': '*bf16', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_convert_element_type_embedding_gt_inductor_lookup_seed_inductor_random_mul_permute_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 33619968}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_convert_element_type_embedding_gt_inductor_lookup_seed_inductor_random_mul_permute_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, load_seed_offset, xnumel, XBLOCK: tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x2 = xindex // 1024 % 16
    x3 = xindex // 16384
    x1 = xindex % 1024
    tmp6 = tl.load(in_ptr1 + (x3 + 512 * x2), None, eviction_policy='evict_last')
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0.to(tl.int32)
    tmp2 = tl.rand(tmp0, tmp1.to(tl.uint32))
    tmp3 = tl.full([1], 0.1, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = tmp4.to(tl.float32)
    tl.device_assert((0 <= tmp6) & (tmp6 < 32000), 'index out of bounds: 0 <= tmp6 < 32000')
    tmp8 = tl.load(in_ptr2 + (x1 + 1024 * tmp6), None)
    tmp9 = tmp5 * tmp8
    tmp10 = tl.full([1], 1.1111111111111112, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr1 + x0, tmp12, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 1048576}, filename=os.path.join(cache_dir(), 'ha', 'cha56slbdcqgncefcx5doxbrvw7gtluuvhvn6x43m4t5tdt2movf.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 8388608}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
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
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((99,), (1,), torch.int64)
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [99], out=buf0)
        assert_size_stride_grouped((arg0_1, arg1_1), ((16, 512), (32000, 1024)), ((512, 1), (1024, 1)), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf2 = empty_strided_cuda((512, 16, 1024), (16384, 1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_convert_element_type_embedding_gt_inductor_lookup_seed_inductor_random_mul_permute_0.run(buf0, arg0_1, arg1_1, buf2, 0, 8388608, stream=raw_stream0)
        del arg0_1
        del arg1_1
        del buf0
        assert_size_stride(arg2_1, (1024, 16, 64), (1024, 64, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf3 = empty_strided_cuda((1024, 16, 64), (1024, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_1.run(arg2_1, buf3, 1048576, stream=raw_stream0)
        del arg2_1
        assert_size_stride_grouped((arg4_1, arg3_1), ((16, 64), (16, 64)), ((64, 1), (64, 1)), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf5 = empty_strided_cuda((512, 16, 16, 64), (16384, 1024, 64, 1), torch.bfloat16)
        buf6 = empty_strided_cuda((512, 16, 16, 64), (16384, 1024, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op4_op5_op6(buf2, buf3, arg4_1, arg3_1, buf5, buf6)
        del arg3_1
        del arg4_1
        del buf2
        del buf3
    return (reinterpret_tensor(buf6, (256, 512, 64), (64, 16384, 1), 0), reinterpret_tensor(buf5, (256, 512, 64), (64, 16384, 1), 0), reinterpret_tensor(buf5, (256, 64, 512), (64, 1, 16384), 0), reinterpret_tensor(buf6, (256, 64, 512), (64, 1, 16384), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [16, 512], 'dtype': 'int64', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 32000}, 'generator': {'kind': 'index', 'low': 0, 'high': 32000}}, {'kind': 'tensor', 'name': None, 'shape': [32000, 1024], 'dtype': 'float32', 'stride': [1024, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1024, 16, 64], 'dtype': 'float32', 'stride': [1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 64], 'dtype': 'float32', 'stride': [64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 64], 'dtype': 'float32', 'stride': [64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1, 8192, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1, 1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 1, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [256, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [256, 512, 64]}]}
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
