"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_mm_18b227377dac
Shape hash: d294fd89
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([16, 128, 256]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
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
def helion_op3_op4(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp4.to(torch.float32)

def call_helion_op3_op4(buf2: torch.Tensor, arg6_1: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op3_op4(reinterpret_tensor(buf2, (4096, 128), (1, 4096), 0), arg6_1, buf4)
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

@triton_heuristics.pointwise(size_hints={'x': 524288}, filename=os.path.join(cache_dir(), 'qw', 'cqwcj4vmlfoi7jxlaz4n42t5ve3uipb6pjrn4cethohxbfvw75od.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_clone_expand_mul_neg_permute_select_slice_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 5521408}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_clone_expand_mul_neg_permute_select_slice_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 256
    x1 = xindex // 256 % 16
    x2 = xindex // 4096
    x3 = xindex // 256
    x4 = xindex
    x5 = xindex % 4096
    tmp39 = tl.load(in_ptr4 + x4, None).to(tl.float32)
    tmp0 = x0.to(tl.int64)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1], 64, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = x0.to(tl.int64)
    tmp5 = tmp4.to(tl.int64)
    tmp6 = tl.full([1], 1, tl.int64)
    tmp7 = tmp5 >= tmp6
    tmp8 = ((-1 + x0) % 2).to(tl.int64)
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = tmp9 == tmp10
    tmp12 = tmp7 & tmp11
    tmp13 = tmp12 & tmp3
    tmp14 = tl.load(in_ptr0 + (2 * triton_helpers.div_floor_integer(-1 + x0, 2) + 256 * x2 + 32768 * x1), tmp13, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (64 * x2 + (-1 + x0) // 2 % 32), tmp13, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp17 = -tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp13, tmp17, tmp18)
    tmp20 = tl.load(in_ptr2 + (x0 + 64 * x3), tmp3, other=0.0).to(tl.float32)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = (x4 % 2).to(tl.int64)
    tmp23 = tmp22.to(tl.int64)
    tmp24 = tmp23 == tmp10
    tmp25 = tmp24 & tmp3
    tmp26 = tl.load(in_ptr0 + (1 + 2 * (x0 // 2) + 256 * x2 + 32768 * x1), tmp25, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (64 * x2 + (1 + 2 * (x5 // 2)) // 2 % 32), tmp25, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp25, tmp28, tmp29)
    tmp31 = tl.where(tmp24, tmp30, tmp20)
    tmp32 = tmp21 + tmp31
    tmp33 = tl.load(in_ptr0 + (x0 + 256 * x2 + 32768 * x1), tmp3, other=0.0).to(tl.float32)
    tmp34 = tl.load(in_ptr3 + (64 * x2 + x0 // 2 % 32), tmp3, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tmp32 + tmp35
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp3, tmp36, tmp37)
    tmp40 = tl.where(tmp3, tmp38, tmp39)
    tmp41 = tmp1 >= tmp2
    tmp42 = tl.load(in_ptr0 + (x0 + 256 * x2 + 32768 * x1), tmp41, other=0.0).to(tl.float32)
    tmp43 = tl.where(tmp41, tmp42, tmp39)
    tmp44 = tmp43 + tmp40
    tl.store(out_ptr1 + x4, tmp44, None)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((16, 128, 128), (16, 128, 256)), ((16384, 128, 1), (256, 4096, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((16, 128, 256), (32768, 256, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1, arg4_1, arg5_1), ((1, 128, 1, 32, 1), (1, 128, 16, 64), (1, 128, 1, 32, 1), (1, 128, 16, 256)), ((8192, 64, 32, 1, 1), (131072, 1024, 64, 1), (8192, 64, 32, 1, 1), (524288, 4096, 256, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        buf2 = empty_strided_cuda((1, 128, 16, 256), (524288, 4096, 256, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_expand_mul_neg_permute_select_slice_view_1.run(buf0, arg2_1, arg3_1, arg4_1, arg5_1, buf2, 524288, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf0
        assert_size_stride(arg6_1, (128, 4096), (4096, 1), 'input')
        arg6_1 = copy_if_misaligned(arg6_1)
        buf4 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op3_op4(buf2, arg6_1, buf4)
        del arg6_1
    return (reinterpret_tensor(buf2, (128, 4096), (4096, 1), 0), buf4)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [16, 128, 128], 'dtype': 'bfloat16', 'stride': [16384, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 128, 256], 'dtype': 'bfloat16', 'stride': [256, 4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128, 1, 32, 1], 'dtype': 'bfloat16', 'stride': [8192, 64, 32, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128, 16, 64], 'dtype': 'bfloat16', 'stride': [131072, 1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128, 1, 32, 1], 'dtype': 'bfloat16', 'stride': [8192, 64, 32, 1, 1], 'device': 'cuda', 'storage_offset': 32}, {'kind': 'tensor', 'name': None, 'shape': [1, 128, 16, 256], 'dtype': 'bfloat16', 'stride': [524288, 4096, 256, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [128, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 128, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [128, 4096]}]}
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
