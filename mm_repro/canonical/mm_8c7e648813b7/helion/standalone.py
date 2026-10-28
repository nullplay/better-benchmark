"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_8c7e648813b7
Shape hash: a109a592
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op2(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 10240]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(2560):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = in0[tile_y, tile_x].to(torch.float32)
        tmp6 = tmp5.to(torch.float32)
        tmp7 = hl.full([], 0.7071067811865476, dtype=torch.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = torch.erf(tmp8)
        tmp10 = hl.full([], 1.0, dtype=torch.float32)
        tmp11 = tmp9 + tmp10
        tmp12 = hl.full([], 0.5, dtype=torch.float32)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp6 * tmp6
        tmp15 = hl.full([], -0.5, dtype=torch.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = torch.exp(tmp16)
        tmp18 = hl.full([], 0.3989422804014327, dtype=torch.float32)
        tmp19 = tmp17 * tmp18
        tmp20 = tmp6 * tmp19
        tmp21 = tmp13 + tmp20
        tmp22 = tmp4 * tmp21
        tmp23 = tmp22.to(torch.float32)
        out0[tile_y.index.reshape([tile_y, 1]) // 128, tile_y.index.reshape([tile_y, 1]) % 128, tile_x.index.reshape([1, tile_x])] = tmp23.to(torch.bfloat16)

def call_helion_op1_op2(buf0: torch.Tensor, arg2_1: torch.Tensor, arg3_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op1_op2(reinterpret_tensor(buf0, (4096, 2560), (2560, 1), 0), arg2_1, arg3_1, buf2)
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

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), 'p5', 'cp5jdrgr3k2qbose4einodbgrs7sxiytctyd7hb34sbfukmdahkr.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_mul_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 2, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 94371840}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_mul_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 10485760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None)
    tmp2 = tl.load(in_ptr1 + x0, None).to(tl.int1)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.full([1], 1.1111111111111112, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tmp1 * tmp5
    tl.store(out_ptr0 + x0, tmp6, None)
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
    assert_size_stride_grouped((arg0_1, arg1_1), ((32, 128, 2560), (32, 128, 2560)), ((327680, 2560, 1), (327680, 2560, 1)), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((32, 128, 2560), (327680, 2560, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_mul_0.run(arg0_1, arg1_1, buf0, 10485760, stream=raw_stream0)
        del arg0_1
        del arg1_1
        assert_size_stride_grouped((arg2_1, arg3_1), ((2560, 10240), (4096, 10240)), ((10240, 1), (10240, 1)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        buf1 = empty_strided_cuda((4096, 10240), (10240, 1), torch.bfloat16)
        buf2 = reinterpret_tensor(buf1, (32, 128, 10240), (1310720, 10240, 1), 0)
        del buf1
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op2(buf0, arg2_1, arg3_1, buf2)
        del arg2_1
        del arg3_1
        del buf0
    return (reinterpret_tensor(buf2, (4096, 10240), (10240, 1), 0), reinterpret_tensor(buf2, (10240, 4096), (1, 10240), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [32, 128, 2560], 'dtype': 'float32', 'stride': [327680, 2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 128, 2560], 'dtype': 'bool', 'stride': [327680, 2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2560, 10240], 'dtype': 'bfloat16', 'stride': [10240, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 10240], 'dtype': 'bfloat16', 'stride': [10240, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [4096, 2560]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 10240]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 10240]}, {'kind': 'shape', 'name': None, 'dims': [4096, 10240]}]}
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
