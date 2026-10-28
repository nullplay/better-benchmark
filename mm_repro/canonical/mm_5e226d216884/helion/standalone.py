"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_5e226d216884
Shape hash: ff165bfc
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 768]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg0_1, arg1_1, buf0)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
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

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'er', 'cer25brcpu7vzb3zqfs4lcv54jzjsvjefwwtl2scavpoeaactlvl.py'), triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_full_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 51118080}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_full_1(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6389760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.full([1], 0.0, tl.float32)
    tl.store(out_ptr0 + x0, tmp0, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'z': 16384, 'y': 16, 'x': 512}, filename=os.path.join(cache_dir(), 'ix', 'cix552wxzsgk2rylr3vbgpbybrtttsgcheybd72jcmqzmegljbdn.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*i64', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'znumel': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'ZBLOCK': 'constexpr', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid3D', 'kernel_name': 'triton_poi_fused_add_clone_convert_element_type_div_exp_full_index_put_mul_permute_slice_sub_view_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': True, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'z': 13369344, 'y': 1769472, 'x': 37080}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_clone_convert_element_type_div_exp_full_index_put_mul_permute_slice_sub_view_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, znumel, ynumel, xnumel, ZBLOCK: tl.constexpr, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    znumel = 12288
    ynumel = 9
    xnumel = 512
    zoffset = tl.program_id(2) * ZBLOCK
    zindex = zoffset + tl.arange(0, ZBLOCK)[:, None, None]
    zmask = tl.full([ZBLOCK], True, tl.int1)[:, None, None]
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex
    z0 = zindex % 384
    z1 = zindex // 384
    z4 = zindex
    tmp0 = tl.load(in_ptr0 + tl.squeeze(x3 + 512 * y2, 0), tl.squeeze(xmask & ymask, 0), eviction_policy='evict_last')[None, :, :]
    tmp6 = tl.load(in_ptr1 + 0)
    tmp7 = tl.broadcast_to(tmp6, [1, 1, 1])
    tmp13 = tl.load(in_ptr2 + tl.squeeze(384 + z0 + 768 * x3 + 393216 * z1, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None, :].to(tl.float32)
    tmp15 = tl.load(in_ptr3 + (y2 + 9 * (z0 // 64) + 54 * x3 + 27648 * z1), xmask & ymask).to(tl.float32)
    tmp16 = tl.load(in_ptr4 + tl.squeeze(y2 + 9 * (z0 // 64), 2), tl.squeeze(ymask, 2), eviction_policy='evict_last')[:, :, None]
    tmp20 = tl.load(in_ptr5 + tl.squeeze(6 * x3 + 3072 * z1 + z0 // 64, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None, :]
    tmp23 = tl.load(in_ptr6 + tl.squeeze(6 * x3 + 3072 * z1 + z0 // 64, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None, :]
    tmp1 = tl.full([1, 1, 1], 520, tl.int32).to(tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 520) | ~(xmask & ymask), 'index out of bounds: 0 <= tmp4 < 520')
    tmp8 = tl.full([1, 1, 1], 1, tl.int32).to(tl.int32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp7 < 0
    tmp11 = tl.where(tmp10, tmp9, tmp7)
    tl.device_assert((0 <= tmp11) & (tmp11 < 1), 'index out of bounds: 0 <= tmp11 < 1')
    tmp14 = tmp13.to(tl.float32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp19 - tmp20
    tmp22 = libdevice.exp(tmp21)
    tmp24 = tmp22 / tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp14 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28.to(tl.float32)
    tl.atomic_add(out_ptr0 + tl.broadcast_to(tmp4 + 520 * z4, [ZBLOCK, YBLOCK, XBLOCK]), tmp29, xmask & ymask, sem='relaxed')
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'mv', 'cmv5ckxqz2uvfvhtsjjkyxnnxqlvoavm5sgtsav6z64aehkmgbw7.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_constant_pad_nd_convert_element_type_permute_squeeze_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 25165824}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_constant_pad_nd_convert_element_type_permute_squeeze_3(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 384 % 512
    x0 = xindex % 384
    x2 = xindex // 196608
    x3 = xindex
    tmp0 = (4 + x1).to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 520, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (4 + x1 + 520 * x0 + 199680 * x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + x3, tmp7, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((16384, 768), (768, 768)), ((768, 1), (768, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((16384, 768), (768, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        buf1 = empty_strided_cuda((32, 384, 520, 1), (199680, 520, 1, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_full_1.run(buf1, 6389760, stream=raw_stream0)
        assert_size_stride_grouped((arg6_1, arg7_1, arg3_1, arg2_1, arg4_1, arg5_1), ((9, 512, 1, 1), (1, 1), (16384, 54), (54,), (98304, 1, 1), (98304, 1, 1)), ((512, 1, 1, 1), (1, 1), (54, 1), (1,), (1, 1, 1), (1, 1, 1)), 'input')
        arg6_1 = copy_if_misaligned(arg6_1)
        arg7_1 = copy_if_misaligned(arg7_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_convert_element_type_div_exp_full_index_put_mul_permute_slice_sub_view_2.run(arg6_1, arg7_1, buf0, arg3_1, arg2_1, arg4_1, arg5_1, buf1, 12288, 9, 512, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del arg7_1
        del buf0
        buf3 = empty_strided_cuda((32, 512, 384), (196608, 384, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_constant_pad_nd_convert_element_type_permute_squeeze_3.run(buf1, buf3, 6291456, stream=raw_stream0)
        del buf1
    return (reinterpret_tensor(buf3, (16384, 384), (384, 1), 0), reinterpret_tensor(buf3, (384, 16384), (1, 384), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [16384, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [768, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [54], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16384, 54], 'dtype': 'bfloat16', 'stride': [54, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [98304, 1, 1], 'dtype': 'float32', 'stride': [1, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [98304, 1, 1], 'dtype': 'float32', 'stride': [1, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [9, 512, 1, 1], 'dtype': 'int64', 'stride': [512, 1, 1, 1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 520}, 'generator': {'kind': 'index', 'low': 0, 'high': 520}}, {'kind': 'tensor', 'name': None, 'shape': [1, 1], 'dtype': 'int64', 'stride': [1, 1], 'device': 'cuda', 'storage_offset': 0, 'gen': {'kind': 'index', 'low': 0, 'high': 1}, 'generator': {'kind': 'index', 'low': 0, 'high': 1}}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 768]}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 12, 64]}, {'kind': 'shape', 'name': None, 'dims': [16384, 384]}, {'kind': 'shape', 'name': None, 'dims': [98304, 64, 1]}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 54]}, {'kind': 'shape', 'name': None, 'dims': [-1, 9, 1]}, {'kind': 'shape', 'name': None, 'dims': [98304, 9, 1]}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 384, 9]}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 3456]}, {'kind': 'shape', 'name': None, 'dims': [32, 384, 9, 1, 512, 1]}, {'kind': 'shape', 'name': None, 'dims': [32, 384, 520, 1]}, {'kind': 'shape', 'name': None, 'dims': [0, 0, -4, -4]}, {'kind': 'shape', 'name': None, 'dims': [16384, 384]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]

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
