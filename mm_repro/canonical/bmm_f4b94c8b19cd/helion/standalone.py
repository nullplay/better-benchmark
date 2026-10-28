"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_f4b94c8b19cd
Shape hash: 88075adf
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([16, 128, 128]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(256):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = B[tile_z, tile_r, tile_x].to(torch.float32)
            acc = torch.baddbmm(acc, tmp0, tmp1)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.float32)

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

@triton_heuristics.persistent_reduction(size_hints={'x': 2048, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'ay', 'caylgel7m3skqmqgtjkiglqxarthi5dpi6fctycaxfw4mpl2bpt4.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_bitwise_and_convert_element_type_div_eq_exp_expand_full_index_iota_le_prepare_softmax_online_sub_unsqueeze_view_where_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 1024, 'r0_': 2098176}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_bitwise_and_convert_element_type_div_eq_exp_expand_full_index_iota_le_prepare_softmax_online_sub_unsqueeze_view_where_1(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 2048
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (r0_2 + 128 * x3), xmask, eviction_policy='evict_first', other=0.0)
    tmp10 = tl.load(in_ptr1 + tl.squeeze(x0, 1), tl.squeeze(xmask, 1), eviction_policy='evict_last')[:, None]
    tmp11 = tl.load(in_ptr1 + tl.squeeze(r0_2, 0), None, eviction_policy='evict_last')[None, :]
    tmp1 = tl.full([1, 1], 0.0625, tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = r0_2.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = x0.to(tl.int64)
    tmp6 = tmp5.to(tl.int64)
    tmp7 = tmp4 <= tmp6
    tmp8 = tl.full([1, 1], True, tl.int1)
    tmp9 = tmp8 & tmp7
    tmp12 = tmp10 == tmp11
    tmp13 = tmp9 & tmp12
    tmp14 = tl.full([1, 1], 0.0, tl.float32)
    tmp15 = tl.full([1, 1], -3.3895313892515355e+38, tl.float32)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp2 + tmp17
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp23 = tl.where(xmask, tmp21, float('-inf'))
    tmp24 = triton_helpers.fmax2(tmp23, 1)[:, None].to(tl.float32)
    tmp25 = tmp19 - tmp24
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(xmask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None].to(tl.float32)
    tmp31 = tmp18 - tmp24
    tmp32 = libdevice.exp(tmp31)
    tmp33 = tmp32 / tmp30
    tmp34 = tmp33.to(tl.float32)
    tl.store(out_ptr2 + (r0_2 + 128 * x3), tmp34, xmask)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg0_1, arg1_1), ((16, 128, 256), (16, 256, 128)), ((256, 4096, 1), (256, 1, 4096)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((16, 128, 128), (16384, 128, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg0_1, arg1_1, buf0)
        del arg0_1
        del arg1_1
        assert_size_stride(arg2_1, (1, 128), (128, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf3 = empty_strided_cuda((1, 16, 128, 128), (262144, 16384, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_bitwise_and_convert_element_type_div_eq_exp_expand_full_index_iota_le_prepare_softmax_online_sub_unsqueeze_view_where_1.run(buf0, arg2_1, buf3, 2048, 128, stream=raw_stream0)
        del arg2_1
        del buf0
    return (reinterpret_tensor(buf3, (16, 128, 128), (16384, 128, 1), 0),)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [16, 128, 256], 'dtype': 'float32', 'stride': [256, 4096, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16, 256, 128], 'dtype': 'float32', 'stride': [256, 1, 4096], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [1, 128], 'dtype': 'int64', 'stride': [128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, -1, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [1, 16, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [16, 128, 128]}]}
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
