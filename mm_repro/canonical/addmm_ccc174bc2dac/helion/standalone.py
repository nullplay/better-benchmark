"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_ccc174bc2dac
Shape hash: 18a356fb
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1000, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(2880):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg2_1: torch.Tensor, arg0_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg2_1, reinterpret_tensor(arg0_1, (2880, 4096), (1, 2880), 0), buf0)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([1000, 512]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(2880):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(arg5_1: torch.Tensor, arg3_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(arg5_1, reinterpret_tensor(arg3_1, (2880, 512), (1, 2880), 0), buf1)
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
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 2097152}, filename=os.path.join(cache_dir(), 'zz', 'czzbt5pqq7cfgoacffisrrihx7xxbrhsw7qklrog2oriwuhvhxpx.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_convert_element_type_cos_expand_mul_permute_sin_split_sub_unsqueeze_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 24584256}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_convert_element_type_cos_expand_mul_permute_sin_split_sub_unsqueeze_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2048000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 32
    x1 = xindex // 32 % 64
    x3 = xindex // 32
    x2 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0 + 64 * x3), None).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + x0, None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (32 + x0 + 64 * x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (32 + x0 + 64 * x3), None).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = x2.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tl_math.cos(tmp10)
    tmp12 = tl.full([1], 1.3465735902799727, tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp5 * tmp14
    tmp17 = tmp1 * tmp16
    tmp19 = tmp1 * tmp18
    tmp20 = tmp17 + tmp19
    tmp21 = tl_math.sin(tmp10)
    tmp22 = tmp21 * tmp12
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp15 - tmp24
    tmp26 = tmp20 * tmp14
    tmp27 = tmp5 * tmp23
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr0 + (x0 + 64 * x2 + 64000 * x1), tmp25, None)
    tl.store(out_ptr1 + (x0 + 64 * x2 + 64000 * x1), tmp28, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 262144}, filename=os.path.join(cache_dir(), 'se', 'cse7fqfedqfdy2pheleoar67jikorhhzenadpxulp7n4kzwpvg3x.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_addmm_convert_element_type_cos_expand_mul_permute_sin_split_sub_unsqueeze_view_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 5, 'num_store': 2, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 3073088}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_addmm_convert_element_type_cos_expand_mul_permute_sin_split_sub_unsqueeze_view_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 256000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = xindex // 32 % 8
    x3 = xindex // 32
    x2 = xindex // 256
    tmp0 = tl.load(in_ptr0 + (x0 + 64 * x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x0 + 64 * x3), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr2 + x0, xmask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr0 + (32 + x0 + 64 * x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (32 + x0 + 64 * x3), xmask).to(tl.float32)
    tmp1 = tl.full([1], 1.0, tl.float32)
    tmp2 = tmp1 * tmp0
    tmp4 = tmp1 * tmp3
    tmp5 = tmp2 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp8 = x2.to(tl.float32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 * tmp9
    tmp11 = tl_math.cos(tmp10)
    tmp12 = tl.full([1], 1.3465735902799727, tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp5 * tmp14
    tmp17 = tmp1 * tmp16
    tmp19 = tmp1 * tmp18
    tmp20 = tmp17 + tmp19
    tmp21 = tl_math.sin(tmp10)
    tmp22 = tmp21 * tmp12
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp20 * tmp23
    tmp25 = tmp15 - tmp24
    tmp26 = tmp20 * tmp14
    tmp27 = tmp5 * tmp23
    tmp28 = tmp26 + tmp27
    tl.store(out_ptr0 + (x0 + 64 * x2 + 64000 * x1), tmp25, xmask)
    tl.store(out_ptr1 + (x0 + 64 * x2 + 64000 * x1), tmp28, xmask)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), 'hs', 'chsayhnotqeam5k6qp5eracxiysyeiitjycnaz3z522f63n6bquh.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_expand_unsqueeze_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 17408000}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_expand_unsqueeze_4(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4096000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 64000
    x2 = xindex // 512000
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 64000 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + x3, tmp0, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride_grouped((arg2_1, arg0_1), ((1000, 2880), (4096, 2880)), ((2880, 1), (2880, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((1000, 4096), (4096, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg2_1, arg0_1, buf0)
        del arg0_1
        del arg2_1
        assert_size_stride_grouped((arg5_1, arg3_1), ((1000, 2880), (512, 2880)), ((2880, 1), (2880, 1)), 'input')
        arg5_1 = copy_if_misaligned(arg5_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((1000, 512), (512, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(arg5_1, arg3_1, buf1)
        del arg3_1
        del arg5_1
        assert_size_stride_grouped((arg1_1, arg6_1), ((4096,), (32,)), ((1,), (1,)), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        buf4 = empty_strided_cuda((1, 64, 1000, 64), (4096000, 64000, 64, 1), torch.bfloat16)
        buf2 = reinterpret_tensor(buf4, (1, 64, 1000, 32), (4096000, 64000, 64, 1), 0)
        buf3 = reinterpret_tensor(buf4, (1, 64, 1000, 32), (4096000, 64000, 64, 1), 32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convert_element_type_cos_expand_mul_permute_sin_split_sub_unsqueeze_view_2.run(arg1_1, buf0, arg6_1, buf2, buf3, 2048000, stream=raw_stream0)
        del arg1_1
        del buf0
        assert_size_stride(arg4_1, (512,), (1,), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        buf7 = empty_strided_cuda((1, 8, 1000, 64), (512000, 64000, 64, 1), torch.bfloat16)
        buf5 = reinterpret_tensor(buf7, (1, 8, 1000, 32), (512000, 64000, 64, 1), 0)
        buf6 = reinterpret_tensor(buf7, (1, 8, 1000, 32), (512000, 64000, 64, 1), 32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_addmm_convert_element_type_cos_expand_mul_permute_sin_split_sub_unsqueeze_view_3.run(arg4_1, buf1, arg6_1, buf5, buf6, 256000, stream=raw_stream0)
        del arg4_1
        del arg6_1
        del buf1
        buf8 = empty_strided_cuda((1, 8, 8, 1000, 64), (4096000, 512000, 64000, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_expand_unsqueeze_4.run(buf7, buf8, 4096000, stream=raw_stream0)
    return (reinterpret_tensor(buf4, (64, 1000, 64), (64000, 64, 1), 0), reinterpret_tensor(buf8, (64, 64, 1000), (64000, 1, 64), 0), reinterpret_tensor(buf7, (1, 8, 127, 64), (512000, 64000, 64, 1), 55872))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [4096, 2880], 'dtype': 'bfloat16', 'stride': [2880, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000, 2880], 'dtype': 'bfloat16', 'stride': [2880, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 2880], 'dtype': 'bfloat16', 'stride': [2880, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1000, 2880], 'dtype': 'bfloat16', 'stride': [2880, 1], 'device': 'cuda', 'storage_offset': 0, 'alias_group': 0}, {'kind': 'tensor', 'name': None, 'shape': [32], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, 4096]}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, -1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, 512]}, {'kind': 'shape', 'name': None, 'dims': [1, 1000, -1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 32, 1]}, {'kind': 'shape', 'name': None, 'dims': [1, 1, 1000]}, {'kind': 'shape', 'name': None, 'dims': [1, 64, 1000, 64]}, {'kind': 'shape', 'name': None, 'dims': [64, 1000, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 8, 8, 1000, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 64, 1000, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 64, 64, 1000]}, {'kind': 'shape', 'name': None, 'dims': [64, 64, 1000]}], 'alias_group_nbytes': [5760000]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6]

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
