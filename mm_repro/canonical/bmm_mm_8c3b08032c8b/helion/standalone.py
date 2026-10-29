"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_mm_8c3b08032c8b
Shape hash: 5626091b
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
def helion_op2_op3(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
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

def call_helion_op2_op3(buf1: torch.Tensor, arg4_1: torch.Tensor, buf3: torch.Tensor) -> None:
    helion_op2_op3(reinterpret_tensor(buf1, (4096, 128), (1, 4096), 0), arg4_1, buf3)
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

@triton_heuristics.pointwise(size_hints={'x': 524288}, filename=os.path.join(cache_dir(), 'x7', 'cx7qxsgdfbpeo3ykiaicpmu2hly7r67sqaekczuxdarr4iacd3tp.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_add_clone_expand_full_mul_neg_permute_select_slice_view_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 4210688}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_add_clone_expand_full_mul_neg_permute_select_slice_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 256
    x3 = xindex
    x4 = xindex // 256
    x1 = xindex // 256 % 128
    x2 = xindex // 32768
    tmp0 = x0.to(tl.int64)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1], 64, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.load(in_ptr0 + x3, tmp3, other=0.0).to(tl.float32)
    tmp5 = tl.full([1], 0.0, tl.float32)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tmp7 = tmp1 < tmp2
    tmp8 = x0.to(tl.int64)
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tl.full([1], 1, tl.int64)
    tmp11 = tmp9 >= tmp10
    tmp12 = ((-1 + x0) % 2).to(tl.int64)
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = tmp13 == tmp14
    tmp16 = tmp11 & tmp15
    tmp17 = tmp16 & tmp7
    tmp18 = tl.load(in_ptr0 + (2 * triton_helpers.div_floor_integer(-1 + x0, 2) + 256 * x4), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp19 = tl.load(in_ptr1 + (64 * x1 + (-1 + x0) // 2 % 32), tmp17, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp20 = tmp18 * tmp19
    tmp21 = -tmp20
    tmp22 = tl.full(tmp21.shape, 0.0, tmp21.dtype)
    tmp23 = tl.where(tmp17, tmp21, tmp22)
    tmp24 = tl.full([1], 0.0, tl.float32)
    tmp25 = tl.where(tmp16, tmp23, tmp24)
    tmp26 = (x3 % 2).to(tl.int64)
    tmp27 = tmp26.to(tl.int64)
    tmp28 = tmp27 == tmp14
    tmp29 = tmp28 & tmp7
    tmp30 = tl.load(in_ptr0 + (1 + 2 * (x0 // 2) + 256 * x4), tmp29, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr1 + (64 * x1 + (1 + 2 * (x0 // 2)) // 2 % 32), tmp29, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp29, tmp32, tmp33)
    tmp35 = tl.where(tmp28, tmp34, tmp24)
    tmp36 = tmp25 + tmp35
    tmp37 = tl.load(in_ptr0 + x3, tmp7, other=0.0).to(tl.float32)
    tmp38 = tl.load(in_ptr2 + (64 * x1 + x0 // 2 % 32), tmp7, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp39 = tmp37 * tmp38
    tmp40 = tmp36 + tmp39
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp7, tmp40, tmp41)
    tmp43 = tl.where(tmp7, tmp42, tmp5)
    tmp44 = tmp6 + tmp43
    tl.store(out_ptr0 + (x0 + 256 * x2 + 4096 * x1), tmp44, None)
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
        assert_size_stride_grouped((arg2_1, arg3_1), ((1, 128, 1, 32, 1), (1, 128, 1, 32, 1)), ((8192, 64, 32, 1, 1), (8192, 64, 32, 1, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((1, 128, 16, 256), (524288, 4096, 256, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_add_clone_expand_full_mul_neg_permute_select_slice_view_1.run(buf0, arg2_1, arg3_1, buf1, 524288, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del buf0
        assert_size_stride(arg4_1, (128, 4096), (4096, 1), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        buf3 = empty_strided_cuda((4096, 4096), (4096, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2_op3(buf1, arg4_1, buf3)
        del arg4_1
    return (reinterpret_tensor(buf1, (128, 4096), (4096, 1), 0), buf3)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [16, 128, 128], 'dtype': 'bfloat16', 'stride': [16384, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 128, 256], 'dtype': 'bfloat16', 'stride': [256, 4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128, 1, 32, 1], 'dtype': 'bfloat16', 'stride': [8192, 64, 32, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128, 1, 32, 1], 'dtype': 'bfloat16', 'stride': [8192, 64, 32, 1, 1], 'device': 'cuda', 'storage_offset': 32}, {'kind': 'tensor', 'name': None, 'shape': [128, 4096], 'dtype': 'bfloat16', 'stride': [4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 128, 256]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 32, 2]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 1, 64]}, {'kind': 'shape', 'name': None, 'dims': [1, 128, 16, 256]}, {'kind': 'shape', 'name': None, 'dims': [128, 4096]}]}
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
