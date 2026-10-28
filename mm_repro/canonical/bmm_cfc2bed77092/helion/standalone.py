"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_cfc2bed77092
Shape hash: d031c658
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([192, 128, 128]):
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

@triton_heuristics.persistent_reduction(size_hints={'x': 32768, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'y3', 'cy3t7cr6kxphf7hyq4pxxfymvfxdpb7rncip6g6xnq2vsfs6k57w.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_convert_element_type_div_embedding_exp_expand_full_iota_le_log_lt_minimum_mul_neg_permute_prepare_softmax_online_sub_unsqueeze_view_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 0, 'r0_': 18874368}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_convert_element_type_div_embedding_exp_expand_full_iota_le_log_lt_minimum_mul_neg_permute_prepare_softmax_online_sub_unsqueeze_view_where_1(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 24576
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
    r0_3 = r0_index
    x4 = xindex
    x0 = xindex % 128
    x1 = xindex // 128 % 6
    tmp0 = tl.load(in_ptr0 + (r0_3 + 128 * x4), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = (-1 * (0 * (0 <= r0_3 + -1 * x0) + (r0_3 + -1 * x0) * (r0_3 + -1 * x0 < 0))).to(tl.int32)
    tmp2 = tl.full([1, 1], 16, tl.int64)
    tmp3 = tmp1 < tmp2
    tmp4 = (-1 * (0 * (0 <= r0_3 + -1 * x0) + (r0_3 + -1 * x0) * (r0_3 + -1 * x0 < 0))).to(tl.float32)
    tmp5 = tl.full([1, 1], 0.0625, tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = tl_math.log(tmp6)
    tmp8 = tl.full([1, 1], 0.48089834696298783, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1, 1], 16.0, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.int64)
    tmp13 = tmp12 + tmp2
    tmp14 = tl.full([1, 1], 31, tl.int64)
    tmp15 = tl.minimum(tmp13, tmp14, tl.PropagateNan.ALL)
    tmp16 = tl.where(tmp3, tmp1, tmp15)
    tl.device_assert((0 <= tmp16) & (tmp16 < 32), 'index out of bounds: 0 <= tmp16 < 32')
    tmp18 = tl.load(in_ptr1 + (x1 + 6 * tmp16), None, eviction_policy='evict_last').to(tl.float32)
    tmp19 = r0_3.to(tl.int64)
    tmp20 = tmp19.to(tl.int64)
    tmp21 = x0.to(tl.int64)
    tmp22 = tmp21.to(tl.int64)
    tmp23 = tmp20 <= tmp22
    tmp24 = tl.full([1, 1], 0.0, tl.float32)
    tmp25 = tl.full([1, 1], -3.3895313892515355e+38, tl.float32)
    tmp26 = tl.where(tmp23, tmp24, tmp25)
    tmp27 = tmp18 + tmp26
    tmp28 = tmp0 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, R0_BLOCK])
    tmp32 = tl.broadcast_to(tmp30, [XBLOCK, R0_BLOCK])
    tmp34 = triton_helpers.fmax2(tmp32, 1)[:, None].to(tl.float32)
    tmp35 = tmp30 - tmp34
    tmp36 = libdevice.exp(tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.sum(tmp37, 1)[:, None].to(tl.float32)
    tmp40 = tmp29 - tmp34
    tmp41 = libdevice.exp(tmp40)
    tmp42 = tmp41 / tmp39
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr3 + (r0_3 + 128 * x4), tmp43, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((192, 128, 64), (192, 64, 128)), ((8192, 64, 1), (8192, 128, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((192, 128, 128), (16384, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride(arg2_1, (32, 6), (6, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf4 = empty_strided_cuda((32, 6, 128, 128), (98304, 16384, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_convert_element_type_div_embedding_exp_expand_full_iota_le_log_lt_minimum_mul_neg_permute_prepare_softmax_online_sub_unsqueeze_view_where_1.run(buf0, arg2_1, buf4, 24576, 128, stream=raw_stream0)
        del arg2_1
        del buf0
    return (reinterpret_tensor(buf4, (192, 128, 128), (16384, 128, 1), 0),)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [192, 128, 64], 'dtype': 'bfloat16', 'stride': [8192, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [192, 64, 128], 'dtype': 'bfloat16', 'stride': [8192, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 6], 'dtype': 'bfloat16', 'stride': [6, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 6, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [128, 128]}, {'kind': 'shape', 'name': None, 'dims': [128, 128]}, {'kind': 'shape', 'name': None, 'dims': [32, -1, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [32, 6, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [192, 128, 128]}]}
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
