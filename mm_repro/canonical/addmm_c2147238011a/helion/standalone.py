"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_c2147238011a
Shape hash: 0068e3c8
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op2(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 3072]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(768):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = tmp4.to(torch.float32)
        tmp6 = hl.full([], 1.0, dtype=torch.float32)
        tmp7 = tmp6 * tmp5
        tmp8 = tmp6 * acc
        tmp9 = tmp7.reshape([tile_x, 1]) + tmp8.permute(1, 0)
        tmp10 = hl.full([], 0.5, dtype=torch.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = tmp11.to(torch.float32)
        tmp13 = tmp9.to(torch.float32)
        tmp14 = tmp13 * tmp13
        tmp15 = tmp14 * tmp13
        tmp16 = hl.full([], 0.044715, dtype=torch.float32)
        tmp17 = tmp15 * tmp16
        tmp18 = tmp13 + tmp17
        tmp19 = hl.full([], 0.7978845608028654, dtype=torch.float32)
        tmp20 = tmp18 * tmp19
        tmp21 = torch.tanh(tmp20)
        tmp22 = tmp21 + tmp6
        tmp23 = tmp12 * tmp22
        tmp24 = tmp23.to(torch.float32)
        out0[tile_y, tile_x] = tmp24.permute(1, 0).to(torch.bfloat16)

def call_helion_op1_op2(arg2_1: torch.Tensor, buf0: torch.Tensor, arg1_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op1_op2(arg2_1, buf0, arg1_1, buf2)
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

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), 'il', 'cilr4vgwfdstsbd53od4qcqy2t2gohnfvxpvm4k2xi24qvg367if.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 18874368}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + x0, tmp1, None)
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
    assert_size_stride(arg0_1, (768, 3072), (3072, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((768, 3072), (3072, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg0_1, buf0, 2359296, stream=raw_stream0)
        del arg0_1
        assert_size_stride_grouped((arg2_1, arg1_1), ((16384, 768), (3072,)), ((768, 1), (1,)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf1 = empty_strided_cuda((16384, 3072), (3072, 1), torch.bfloat16)
        buf2 = buf1
        del buf1
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op2(arg2_1, buf0, arg1_1, buf2)
        del arg1_1
        del arg2_1
        del buf0
    return (buf2, reinterpret_tensor(buf2, (3072, 16384), (1, 3072), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [768, 3072], 'dtype': 'float32', 'stride': [3072, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [3072], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [16384, 768], 'dtype': 'bfloat16', 'stride': [768, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 512, 3072]}, {'kind': 'shape', 'name': None, 'dims': [-1, 3072]}]}
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
