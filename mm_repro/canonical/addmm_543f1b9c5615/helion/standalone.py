"""First-shape Helion native MM wrapper; generated without execution.
Repro: addmm_543f1b9c5615
Shape hash: 02c43fde
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op4_op7_op2_op5_op8(A: torch.Tensor, B: torch.Tensor, B1: torch.Tensor, B2: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, in2: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 2560]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc2 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(2560):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = B1[tile_r, tile_x].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp1, tmp5)
            tmp6 = B2[tile_r, tile_x].to(torch.float32)
            tmp7 = tmp6.to(torch.bfloat16)
            acc2 = torch.addmm(acc2, tmp1, tmp7)
        tmp8 = in0[tile_x].to(torch.float32)
        tmp9 = tmp8.to(torch.float32)
        tmp10 = hl.full([], 1.0, dtype=torch.float32)
        tmp11 = tmp10 * tmp9
        tmp12 = tmp10 * acc
        tmp13 = tmp11.reshape([tile_x, 1]) + tmp12.permute(1, 0)
        out0[tile_y, tile_x] = tmp13.permute(1, 0).to(torch.bfloat16)
        tmp14 = in1[tile_x].to(torch.float32)
        tmp15 = tmp14.to(torch.float32)
        tmp16 = hl.full([], 1.0, dtype=torch.float32)
        tmp17 = tmp16 * tmp15
        tmp18 = tmp16 * acc1
        tmp19 = tmp17.reshape([tile_x, 1]) + tmp18.permute(1, 0)
        out1[tile_y, tile_x] = tmp19.permute(1, 0).to(torch.bfloat16)
        tmp20 = in2[tile_x].to(torch.float32)
        tmp21 = tmp20.to(torch.float32)
        tmp22 = hl.full([], 1.0, dtype=torch.float32)
        tmp23 = tmp22 * tmp21
        tmp24 = tmp22 * acc2
        tmp25 = tmp23.reshape([tile_x, 1]) + tmp24.permute(1, 0)
        out2[tile_y, tile_x] = tmp25.permute(1, 0).to(torch.bfloat16)

def call_helion_op1_op4_op7_op2_op5_op8(arg2_1: torch.Tensor, buf0: torch.Tensor, buf3: torch.Tensor, buf6: torch.Tensor, arg0_1: torch.Tensor, arg3_1: torch.Tensor, arg5_1: torch.Tensor, buf2: torch.Tensor, buf5: torch.Tensor, buf8: torch.Tensor) -> None:
    helion_op1_op4_op7_op2_op5_op8(arg2_1, reinterpret_tensor(buf0, (2560, 2560), (1, 2560), 0), reinterpret_tensor(buf3, (2560, 2560), (1, 2560), 0), reinterpret_tensor(buf6, (2560, 2560), (1, 2560), 0), arg0_1, arg3_1, arg5_1, buf2, buf5, buf8)
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

@triton_heuristics.pointwise(size_hints={'x': 8388608}, filename=os.path.join(cache_dir(), 'zi', 'czidzftb4ns73tqzh63ib2vaypf3f5ykcmb65opug4adhiku5sp7.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 52428800}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 6553600
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    assert_size_stride(arg1_1, (2560, 2560), (2560, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf0 = empty_strided_cuda((2560, 2560), (2560, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg1_1, buf0, 6553600, stream=raw_stream0)
        del arg1_1
        assert_size_stride(arg4_1, (2560, 2560), (2560, 1), 'input')
        arg4_1 = copy_if_misaligned(arg4_1)
        buf3 = empty_strided_cuda((2560, 2560), (2560, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg4_1, buf3, 6553600, stream=raw_stream0)
        del arg4_1
        assert_size_stride(arg6_1, (2560, 2560), (2560, 1), 'input')
        arg6_1 = copy_if_misaligned(arg6_1)
        buf6 = empty_strided_cuda((2560, 2560), (2560, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg6_1, buf6, 6553600, stream=raw_stream0)
        del arg6_1
        assert_size_stride_grouped((arg2_1, arg0_1, arg3_1, arg5_1), ((4096, 2560), (2560,), (2560,), (2560,)), ((2560, 1), (1,), (1,), (1,)), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        arg0_1 = copy_if_misaligned(arg0_1)
        arg3_1 = copy_if_misaligned(arg3_1)
        arg5_1 = copy_if_misaligned(arg5_1)
        buf1 = empty_strided_cuda((4096, 2560), (2560, 1), torch.bfloat16)
        buf4 = empty_strided_cuda((4096, 2560), (2560, 1), torch.bfloat16)
        buf7 = empty_strided_cuda((4096, 2560), (2560, 1), torch.bfloat16)
        buf2 = buf1
        del buf1
        buf5 = buf4
        del buf4
        buf8 = buf7
        del buf7
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op4_op7_op2_op5_op8(arg2_1, buf0, buf3, buf6, arg0_1, arg3_1, arg5_1, buf2, buf5, buf8)
        del arg0_1
        del arg2_1
        del arg3_1
        del arg5_1
        del buf0
        del buf3
        del buf6
    return (reinterpret_tensor(buf2, (32, 32, 128, 80), (327680, 80, 2560, 1), 0), reinterpret_tensor(buf5, (32, 32, 128, 80), (327680, 80, 2560, 1), 0), reinterpret_tensor(buf8, (32, 32, 128, 80), (327680, 80, 2560, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [2560], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2560, 2560], 'dtype': 'float32', 'stride': [2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [4096, 2560], 'dtype': 'bfloat16', 'stride': [2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2560], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2560, 2560], 'dtype': 'float32', 'stride': [2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2560], 'dtype': 'float32', 'stride': [1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [2560, 2560], 'dtype': 'float32', 'stride': [2560, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 2560]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, -1, 80]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 2560]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, 2560]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, -1, 80]}, {'kind': 'shape', 'name': None, 'dims': [32, 128, -1, 80]}]}
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
