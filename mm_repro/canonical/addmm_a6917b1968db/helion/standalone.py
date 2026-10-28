"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_a6917b1968db
Shape hash: 45a8af0f
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op4(A: torch.Tensor, B: torch.Tensor, B1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(4096):
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
    helion_op1_op4(arg2_1, reinterpret_tensor(buf0, (4096, 4096), (1, 4096), 0), reinterpret_tensor(buf3, (4096, 4096), (1, 4096), 0), buf1, buf4)
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

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), 'tb', 'ctb72z7waz54thqiq5dxb55syxvjwzp5dn2wdcdknawl5q2ncpit.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 134217728}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
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

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), '4o', 'c4onfkscz5ru6fgps22jhvs4j4x6agnx3cf5vanvmimik73cmsmw.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_addmm_clone_convert_element_type_mul_permute_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 100679680}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_clone_convert_element_type_mul_permute_view_2(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 64
    x2 = xindex // 32768 % 64
    x1 = xindex // 64 % 512
    x3 = xindex // 2097152
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x2), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0 + 64 * x2 + 4096 * x1 + 2097152 * x3), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1], 1.0, tl.float32)
    tmp3 = tmp2 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = tmp3 + tmp5
    tmp7 = tl.full([1], 0.3535533905932738, tl.float32)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + x4, tmp8, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'y': 32768, 'x': 512}, tile_hint=TileHint.DEFAULT, filename=os.path.join(cache_dir(), 'k4', 'ck4zn6hygtb5bozemmr5abib5bhma34qghm6qu2nwoiq64twuu63.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_addmm_clone_convert_element_type_mul_permute_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 33554432, 'x': 67125248}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_addmm_clone_convert_element_type_mul_permute_view_3(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 32768
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    y0 = yindex % 4096
    x2 = xindex
    y1 = yindex // 4096
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + tl.squeeze(y0, 1), None, eviction_policy='evict_last')[:, None]
    tmp4 = tl.load(in_ptr1 + (y0 + 4096 * x2 + 2097152 * y1), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.full([1, 1], 1.0, tl.float32)
    tmp3 = tmp2 * tmp1
    tmp5 = tmp2 * tmp4
    tmp6 = tmp3 + tmp5
    tmp7 = tl.full([1, 1], 0.3535533905932738, tl.float32)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2 + 512 * y3), tmp8, xmask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg1_1, (4096, 4096), (4096, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg1_1, buf0, 16777216, stream=raw_stream0)
        del arg1_1
        assert_size_stride(arg4_1, (4096, 4096), (4096, 1), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        buf3 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg4_1, buf3, 16777216, stream=raw_stream0)
        del arg4_1
        assert_size_stride(arg2_1, (4096, 4096), (4096, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf1 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((4096, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op4(arg2_1, buf0, buf3, buf1, buf4)
        del arg2_1
        del buf0
        assert_size_stride(arg0_1, (4096,), (1,), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf2 = reinterpret_tensor(buf3, (8, 64, 512, 64), (2097152, 32768, 64, 1), 0)
        del buf3
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_convert_element_type_mul_permute_view_2.run(arg0_1, buf1, buf2, 16777216, stream=raw_stream0)
        del arg0_1
        assert_size_stride(arg3_1, (4096,), (1,), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf5 = reinterpret_tensor(buf1, (8, 64, 64, 512), (2097152, 32768, 512, 1), 0)
        del buf1
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_clone_convert_element_type_mul_permute_view_3.run(arg3_1, buf4, buf5, 32768, 512, stream=raw_stream0)
        del arg3_1
        del buf4
    return (reinterpret_tensor(buf2, (512, 512, 64), (32768, 64, 1), 0), reinterpret_tensor(buf5, (512, 64, 512), (32768, 512, 1), 0), reinterpret_tensor(buf2, (512, 64, 512), (32768, 1, 64), 0), reinterpret_tensor(buf5, (512, 512, 64), (32768, 1, 512), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'float32', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 4096], 'dtype': 'float32', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 4096]}, {'kind': 'shape', 'name': None, 'dims': [8, 512, -1, 64]}, {'kind': 'shape', 'name': None, 'dims': [8, 512, 4096]}, {'kind': 'shape', 'name': None, 'dims': [8, 512, -1, 64]}, {'kind': 'shape', 'name': None, 'dims': [8, 64, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [512, 512, 64]}, {'kind': 'shape', 'name': None, 'dims': [8, 64, 64, 512]}, {'kind': 'shape', 'name': None, 'dims': [512, 64, 512]}]}
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
