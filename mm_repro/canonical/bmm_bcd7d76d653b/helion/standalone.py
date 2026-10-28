"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_bcd7d76d653b
Shape hash: d40dc1d8
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([512, 128, 128]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
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
def helion_op5_op6(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([512, 64, 128]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(128):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z.index.reshape([tile_z, 1, 1]) // 16, (tile_x.index.reshape([1, 1, tile_x]) + 128 * tile_y.index.reshape([1, tile_y, 1]) + 8192 * tile_z.index.reshape([tile_z, 1, 1])) % 128, (tile_y.index.reshape([1, tile_y, 1]) + 64 * tile_z.index.reshape([tile_z, 1, 1])) % 1024] = acc.to(torch.bfloat16)

def call_helion_op5_op6(arg7_1: torch.Tensor, buf4: torch.Tensor, buf6: torch.Tensor) -> None:
    helion_op5_op6(arg7_1, reinterpret_tensor(buf4, (512, 128, 128), (16384, 128, 1), 0), buf6)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 65536, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'gf', 'cgfkeh6qszym2g2jou3hwtzrnqjpmdhi4ovhuwd4bxc4cjxypgkz.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*i1', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_convert_element_type_div_eq_exp_fma_full_lt_maximum_mul_neg_sub_sum_view_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 7, 'num_store': 1, 'num_reduction': 1, 'autotune_hints': (), 'tiling_scores': {'x': 524288, 'r0_': 77594624}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_convert_element_type_div_eq_exp_fma_full_lt_maximum_mul_neg_sub_sum_view_where_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = xindex % 128
    x1 = xindex // 128
    x5 = xindex // 2048
    tmp0 = tl.load(in_ptr0 + (r0_2 + 128 * x3), None, eviction_policy='evict_first').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 128 * x3), None, eviction_policy='evict_first').to(tl.int1)
    tmp7 = tl.load(in_ptr2 + (r0_2 + 128 * x3), None, eviction_policy='evict_first').to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r0_2 + 128 * x0 + 16384 * (x1 // 16)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + tl.squeeze(x3, 1), None, eviction_policy='evict_last')[:, None]
    tmp16 = tl.load(in_ptr5 + tl.squeeze(x3, 1), None, eviction_policy='evict_last')[:, None]
    tmp24 = tl.load(in_ptr3 + (r0_2 + 128 * x0 + 16384 * x5), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.full([1, 1], 1.1111111111111112, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
    tmp12 = tl.maximum(tmp10, tmp11, tl.PropagateNan.ALL)
    tmp14 = tmp12 - tmp13
    tmp15 = libdevice.exp(tmp14)
    tmp17 = tmp15 / tmp16
    tmp18 = tmp6 * tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp22 = -tmp17
    tmp23 = tl.fma(tmp22, tmp21, tmp18)
    tmp25 = tmp8 + tmp24
    tmp26 = tmp25 < tmp11
    tmp27 = tmp25 == tmp11
    tmp28 = tl.full([1, 1], 0.5, tl.float32)
    tmp29 = tmp23 * tmp28
    tmp30 = tl.where(tmp27, tmp29, tmp23)
    tmp31 = tl.full([1, 1], 0.0, tl.float32)
    tmp32 = tl.where(tmp26, tmp31, tmp30)
    tmp33 = tmp32.to(tl.float32)
    tl.store(out_ptr1 + (r0_2 + 128 * x3), tmp33, None)
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((512, 128, 64), (512, 64, 128)), ((8192, 64, 1), (8192, 1, 64)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((512, 128, 128), (16384, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1, arg4_1, arg5_1, arg6_1), ((512, 128, 128), (512, 128, 128), (32, 1, 128, 128), (512, 128, 1), (512, 128, 1)), ((16384, 128, 1), (16384, 128, 1), (16384, 16384, 128, 1), (128, 1, 1), (128, 1, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg4_1 = copy_if_misaligned(arg4_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        arg6_1 = copy_if_misaligned(arg6_1)
        buf4 = empty_strided_cuda((32, 16, 128, 128), (262144, 16384, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_convert_element_type_div_eq_exp_fma_full_lt_maximum_mul_neg_sub_sum_view_where_1.run(buf0, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, buf4, 65536, 128, stream=raw_stream0)
        del arg2_1
        del arg3_1
        del arg4_1
        del arg5_1
        del arg6_1
        del buf0
        assert_size_stride(arg7_1, (512, 64, 128), (8192, 1, 64), 'input')
        arg7_1 = copy_if_misaligned(arg7_1)
        buf6 = empty_strided_cuda((32, 128, 1024), (131072, 1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op5_op6(arg7_1, buf4, buf6)
        del arg7_1
    return (reinterpret_tensor(buf4, (512, 128, 128), (16384, 128, 1), 0), reinterpret_tensor(buf6, (4096, 1024), (1024, 1), 0), reinterpret_tensor(buf6, (1024, 4096), (1, 1024), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [512, 128, 64], 'dtype': 'bfloat16', 'stride': [8192, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 64, 128], 'dtype': 'bfloat16', 'stride': [8192, 1, 64], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 128, 128], 'dtype': 'bool', 'stride': [16384, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 128, 128], 'dtype': 'bfloat16', 'stride': [16384, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 1, 128, 128], 'dtype': 'float32', 'stride': [16384, 16384, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 128, 1], 'dtype': 'float32', 'stride': [128, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 128, 1], 'dtype': 'float32', 'stride': [128, 1, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 64, 128], 'dtype': 'bfloat16', 'stride': [8192, 1, 64], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 16, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [512, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [32, 16, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [512, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [32, 16, 128, 64]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 1024]}, {'kind': 'shape', 'name': None, 'dims': [4096, 1024]}]}
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
