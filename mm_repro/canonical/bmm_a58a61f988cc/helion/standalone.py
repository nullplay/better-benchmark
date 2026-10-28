"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_a58a61f988cc
Shape hash: ab463ce3
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op0(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([512, 128, 128]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(80):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op0(arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf0: torch.Tensor) -> None:
    helion_op0(arg1_1, arg2_1, buf0)
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

@triton_heuristics.persistent_reduction(size_hints={'x': 65536, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'fq', 'cfqnuh7m6mmaqvcps6uwzyvdcfwcaemwrjr54wap26eahifjev3b.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*i1', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_any_convert_element_type_div_eq_exp_full_logical_not_prepare_softmax_online_sub_view_where_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 5, 'autotune_hints': (), 'tiling_scores': {'x': 128, 'r0_': 50331648}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_any_convert_element_type_div_eq_exp_full_logical_not_prepare_softmax_online_sub_view_where_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 128 * x3), None, eviction_policy='evict_first').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + tl.squeeze(x0, 1), None, eviction_policy='evict_last')[:, None].to(tl.int1)
    tmp2 = tl.full([1, 1], 0.0, tl.float32)
    tmp3 = tl.full([1, 1], float('-inf'), tl.float32)
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tmp5 = tmp0 + tmp4
    tmp6 = tmp5 == tmp3
    tmp7 = tmp6 == 0
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tmp8 != 0
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = triton_helpers.any(tmp10, 1)[:, None].to(tl.int1)
    tmp13 = tmp5.to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp18 = triton_helpers.fmax2(tmp16, 1)[:, None].to(tl.float32)
    tmp19 = tmp14 - tmp18
    tmp20 = libdevice.exp(tmp19)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, R0_BLOCK])
    tmp23 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp24 = tmp12 == 0
    tmp25 = tmp13 - tmp18
    tmp26 = libdevice.exp(tmp25)
    tmp27 = tmp26 / tmp23
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl.where(tmp24, tmp2, tmp28)
    tl.store(in_out_ptr0 + (r0_2 + 128 * x3), tmp29, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride_grouped((arg1_1, arg2_1), ((512, 128, 80), (512, 80, 128)), ((10240, 80, 1), (10240, 128, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf0 = empty_strided_cuda((512, 128, 128), (16384, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op0(arg1_1, arg2_1, buf0)
        del arg1_1
        del arg2_1
        assert_size_stride(arg0_1, (16, 1, 128, 128), (0, 128, 1, 0), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf4 = reinterpret_tensor(buf0, (16, 32, 128, 128), (524288, 16384, 128, 1), 0)
        del buf0
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_any_convert_element_type_div_eq_exp_full_logical_not_prepare_softmax_online_sub_view_where_1.run(buf4, arg0_1, 65536, 128, stream=raw_stream0)
        del arg0_1
    return (buf4, reinterpret_tensor(buf4, (512, 128, 128), (16384, 128, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [16, 1, 128, 128], 'dtype': 'bool', 'stride': [0, 128, 1, 0], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 128, 80], 'dtype': 'bfloat16', 'stride': [10240, 80, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 80, 128], 'dtype': 'bfloat16', 'stride': [10240, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [16, 32, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [16, 32, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [16, 32, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [512, 128, 128]}]}
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
