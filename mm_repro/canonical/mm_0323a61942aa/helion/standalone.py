"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_0323a61942aa
Shape hash: 8da39bde
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([720, 240]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(8192):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        out0[tile_y, tile_x] = tmp4.to(torch.float32)

def call_helion_op1_op2(buf0: torch.Tensor, arg3_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op1_op2(reinterpret_tensor(buf0, (720, 8192), (1, 720), 0), arg3_1, buf2)
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

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'cu', 'ccurxt2ykxqlkxoto5vr5tl75x7vfcn6lgd2trs4zresxhvokpq2.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_cat_clone_constant_pad_nd_permute_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 3, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 36175872}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_cat_clone_constant_pad_nd_permute_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 5898240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x6 = xindex // 3840
    x0 = xindex % 60
    x3 = xindex // 3840 % 512
    x4 = xindex // 1966080
    x7 = xindex // 60 % 64
    x1 = xindex // 60 % 16
    x2 = xindex // 960 % 4
    tmp0 = x6.to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x6.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1], 512, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = x0.to(tl.int32)
    tmp8 = tl.full([1], 64, tl.int64)
    tmp9 = tmp7 < tmp8
    tmp10 = tmp9 & tmp6
    tmp11 = tl.load(in_ptr0 + (x0 + 64 * x7 + 4096 * (x3 + 512 * x4)), tmp10, other=0.0).to(tl.float32)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp6, tmp11, tmp12)
    tmp14 = tmp4 >= tmp5
    tmp15 = tl.full([1], 1024, tl.int64)
    tmp16 = tmp4 < tmp15
    tmp17 = tmp14 & tmp16
    tmp18 = x0.to(tl.int32)
    tmp19 = tl.full([1], 64, tl.int64)
    tmp20 = tmp18 < tmp19
    tmp21 = tmp20 & tmp17
    tmp22 = tl.load(in_ptr1 + (x0 + 64 * x7 + 4096 * (-512 + x3 + 512 * x4)), tmp21, other=0.0).to(tl.float32)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp17, tmp22, tmp23)
    tmp25 = tmp0 >= tmp15
    tmp26 = tl.full([1], 1536, tl.int64)
    tmp27 = tmp0 < tmp26
    tmp28 = x0.to(tl.int32)
    tmp29 = tl.full([1], 64, tl.int64)
    tmp30 = tmp28 < tmp29
    tmp31 = tmp30 & tmp25
    tmp32 = tl.load(in_ptr2 + (x0 + 64 * x7 + 4096 * (-1024 + x3 + 512 * x4)), tmp31, other=0.0).to(tl.float32)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp25, tmp32, tmp33)
    tmp35 = tl.where(tmp17, tmp24, tmp34)
    tmp36 = tl.where(tmp6, tmp13, tmp35)
    tl.store(out_ptr0 + (x0 + 60 * x2 + 240 * x4 + 720 * x1 + 11520 * x3), tmp36, None)
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
    arg0_1, arg1_1, arg2_1, arg3_1 = args
    args.clear()
    assert_size_stride_grouped((arg2_1, arg1_1, arg0_1), ((512, 4, 16, 64), (512, 4, 16, 64), (512, 4, 16, 64)), ((4096, 1024, 64, 1), (4096, 1024, 64, 1), (4096, 1024, 64, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg2_1 = copy_if_misaligned(arg2_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((512, 16, 3, 4, 60), (11520, 720, 240, 60, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_cat_clone_constant_pad_nd_permute_view_0.run(arg2_1, arg1_1, arg0_1, buf0, 5898240, stream=raw_stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        assert_size_stride(arg3_1, (8192, 240), (240, 1), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf2 = empty_strided_cuda((720, 240), (240, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op2(buf0, arg3_1, buf2)
        del arg3_1
        del buf0
    return (buf2,)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [512, 4, 16, 64], 'dtype': 'bfloat16', 'stride': [4096, 1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 4, 16, 64], 'dtype': 'bfloat16', 'stride': [4096, 1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 4, 16, 64], 'dtype': 'bfloat16', 'stride': [4096, 1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 240], 'dtype': 'bfloat16', 'stride': [240, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [0, -4]}, {'kind': 'shape', 'name': None, 'dims': [0, -4]}, {'kind': 'shape', 'name': None, 'dims': [0, -4]}, {'kind': 'shape', 'name': None, 'dims': [3, 512, 4, 16, 60]}, {'kind': 'shape', 'name': None, 'dims': [512, 16, 720]}, {'kind': 'shape', 'name': None, 'dims': [8192, 720]}]}
COMPILED_INPUT_ORDER = [0, 1, 2, 3]

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
