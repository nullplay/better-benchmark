"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_73d472a0753f
Shape hash: 3b842c3a
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2_op3(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 4096]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(9216):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        tmp9 = torch.relu(tmp8)
        out0[tile_y, tile_x] = tmp9.permute(1, 0).to(torch.bfloat16)

def call_helion_op2_op3(buf1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf3: torch.Tensor) -> None:
    helion_op2_op3(reinterpret_tensor(buf1, (128, 9216), (9216, 1), 0), reinterpret_tensor(arg1_1, (9216, 4096), (1, 9216), 0), arg2_1, buf3)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_size_stride_grouped = torch._C._dynamo.guards.assert_size_stride_grouped
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
async_compile = AsyncCompile()
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 2097152}, filename=os.path.join(cache_dir(), 'ds', 'cdsod7jfsioa62imkufk3c3y53qavugtx5vtywplbytzwbp6zrq4.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d__low_memory_max_pool_with_offsets_relu_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 9, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 4718592}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused__adaptive_avg_pool2d__low_memory_max_pool_with_offsets_relu_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 6
    x1 = xindex // 6 % 6
    x2 = xindex // 36
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr0 + (1 + 2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr0 + (2 + 2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr0 + (13 + 2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr0 + (14 + 2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr0 + (15 + 2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (26 + 2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr0 + (27 + 2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr0 + (28 + 2 * x0 + 26 * x1 + 169 * x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.full([1], 0, tl.int32)
    tmp2 = tl.maximum(tmp1, tmp0, tl.PropagateNan.ALL)
    tmp4 = tl.maximum(tmp1, tmp3, tl.PropagateNan.ALL)
    tmp5 = tl.maximum(tmp2, tmp4, tl.PropagateNan.ALL)
    tmp7 = tl.maximum(tmp1, tmp6, tl.PropagateNan.ALL)
    tmp8 = tl.maximum(tmp5, tmp7, tl.PropagateNan.ALL)
    tmp10 = tl.maximum(tmp1, tmp9, tl.PropagateNan.ALL)
    tmp11 = tl.maximum(tmp8, tmp10, tl.PropagateNan.ALL)
    tmp13 = tl.maximum(tmp1, tmp12, tl.PropagateNan.ALL)
    tmp14 = tl.maximum(tmp11, tmp13, tl.PropagateNan.ALL)
    tmp16 = tl.maximum(tmp1, tmp15, tl.PropagateNan.ALL)
    tmp17 = tl.maximum(tmp14, tmp16, tl.PropagateNan.ALL)
    tmp19 = tl.maximum(tmp1, tmp18, tl.PropagateNan.ALL)
    tmp20 = tl.maximum(tmp17, tmp19, tl.PropagateNan.ALL)
    tmp22 = tl.maximum(tmp1, tmp21, tl.PropagateNan.ALL)
    tmp23 = tl.maximum(tmp20, tmp22, tl.PropagateNan.ALL)
    tmp25 = tl.maximum(tmp1, tmp24, tl.PropagateNan.ALL)
    tmp26 = tl.maximum(tmp23, tmp25, tl.PropagateNan.ALL)
    tl.store(in_out_ptr0 + x3, tmp26, None)
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
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, 256, 13, 13), (43264, 169, 13, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((128, 256, 6, 6), (9216, 36, 6, 1), torch.bfloat16)
        buf1 = buf0
        del buf0
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused__adaptive_avg_pool2d__low_memory_max_pool_with_offsets_relu_0.run(buf1, arg0_1, 1179648, stream=raw_stream0)
        del arg0_1
        assert_size_stride_grouped((arg1_1, arg2_1), ((4096, 9216), (4096,)), ((9216, 1), (1,)), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf2 = empty_strided_cuda((128, 4096), (4096, 1), torch.bfloat16)
        buf3 = buf2
        del buf2
        raw_stream0 = get_raw_stream(0)
        call_helion_op2_op3(buf1, arg1_1, arg2_1, buf3)
        del arg1_1
        del arg2_1
        del buf1
    return (buf3,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [128, 256, 13, 13], 'dtype': 'bfloat16', 'stride': [43264, 169, 13, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 9216], 'dtype': 'bfloat16', 'stride': [9216, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096], 'dtype': 'bfloat16', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [3, 3]}, {'kind': 'shape', 'name': None, 'dims': [2, 2]}, {'kind': 'shape', 'name': None, 'dims': [128, 9216]}]}
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
