"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_mm_21a6c25f303d
Shape hash: 908f3e93
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op2_op3(A: torch.Tensor, B: torch.Tensor, B1: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([8192, 512]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        acc1 = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
            tmp4 = B1[tile_r, tile_x].to(torch.float32)
            tmp5 = tmp4.to(torch.bfloat16)
            acc1 = torch.addmm(acc1, tmp1, tmp5)
        out0[tile_y, tile_x] = acc1.to(torch.bfloat16)
        out1[tile_y.index.reshape([tile_y, 1]) // 1024, (tile_x.index.reshape([1, tile_x]) + 512 * tile_y.index.reshape([tile_y, 1])) // 64 % 8, tile_y.index.reshape([tile_y, 1]) % 1024, (tile_x.index.reshape([1, tile_x]) + 512 * tile_y.index.reshape([tile_y, 1])) % 64] = acc.to(torch.bfloat16)

def call_helion_op1_op2_op3(arg1_1: torch.Tensor, buf0: torch.Tensor, arg2_1: torch.Tensor, buf2: torch.Tensor, buf3: torch.Tensor) -> None:
    helion_op1_op2_op3(arg1_1, buf0, arg2_1, buf2, buf3)
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op5_op6_op7(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor, out1: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([64, 1024, 1024]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = in0[tile_z.index.reshape([tile_z, 1, 1]) // 8, tile_z.index.reshape([tile_z, 1, 1]) % 8, tile_y.index.reshape([1, tile_y, 1]), tile_x.index.reshape([1, 1, tile_x])].to(torch.float32)
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(torch.float32)
        out0[tile_z, tile_y, tile_x] = tmp7.to(torch.bfloat16)
        tmp8 = tmp7.to(torch.float32)
        out1[tile_z, tile_y, tile_x] = tmp8.to(torch.float32)

def call_helion_op5_op6_op7(buf3: torch.Tensor, buf4: torch.Tensor, arg3_1: torch.Tensor, buf6: torch.Tensor, buf7: torch.Tensor) -> None:
    helion_op5_op6_op7(reinterpret_tensor(buf3, (64, 1024, 64), (65536, 64, 1), 0), reinterpret_tensor(buf4, (64, 64, 1024), (65536, 1024, 1), 0), arg3_1, reinterpret_tensor(buf6, (64, 1024, 1024), (1048576, 1024, 1), 0), reinterpret_tensor(buf7, (64, 1024, 1024), (1048576, 1024, 1), 0))
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

@triton_heuristics.pointwise(size_hints={'y': 512, 'x': 512}, tile_hint=TileHint.SQUARE, filename=os.path.join(cache_dir(), 'wc', 'cwcu6snd6kri4ruj33jkhskqxcxdyygmqsftx2ru46lgarknvsvw.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid2D', 'kernel_name': 'triton_poi_fused_convert_element_type_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'y': 1048576, 'x': 1048576}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 512
    xnumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512 * x1), xmask & ymask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x1 + 512 * y0), tmp1, xmask & ymask)
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

@triton_heuristics.pointwise(size_hints={'x': 4194304}, filename=os.path.join(cache_dir(), 'w2', 'cw2kipwfndoktvfr4t7ncelpzamzud5mcnucscjcbcl22ktko6ah.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_clone_permute_view_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 8388608}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_clone_permute_view_2(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, None).to(tl.float32)
    tl.store(out_ptr0 + (1024 * (x0 % 512) + 524288 * (x0 // 524288) + x0 // 512 % 1024), tmp0, None)
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
    assert_size_stride(arg0_1, (512, 512), (1, 512), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((512, 512), (512, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_0.run(arg0_1, buf0, 512, 512, stream=raw_stream0)
        del arg0_1
        assert_size_stride_grouped((arg1_1, arg2_1), ((8192, 512), (512, 512)), ((512, 1), (1, 512)), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        arg2_1 = copy_if_misaligned(arg2_1)
        buf2 = empty_strided_cuda((8192, 512), (512, 1), torch.bfloat16)
        buf3 = empty_strided_cuda((8, 8, 1024, 64), (524288, 65536, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op2_op3(arg1_1, buf0, arg2_1, buf2, buf3)
        del arg1_1
        del arg2_1
        del buf0
        buf4 = empty_strided_cuda((8, 8, 64, 1024), (524288, 65536, 1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_clone_permute_view_2.run(buf2, buf4, 4194304, stream=raw_stream0)
        del buf2
        assert_size_stride(arg3_1, (8, 8, 1024, 1024), (8388608, 1, 8192, 8), 'input')
        arg3_1 = copy_if_misaligned(arg3_1)
        buf5 = empty_strided_cuda((64, 1024, 1024), (1048576, 1024, 1), torch.bfloat16)
        buf6 = reinterpret_tensor(buf5, (8, 8, 1024, 1024), (8388608, 1048576, 1024, 1), 0)
        del buf5
        buf7 = empty_strided_cuda((8, 8, 1024, 1024), (8388608, 1048576, 1024, 1), torch.float32)
        raw_stream0 = get_raw_stream(0)
        call_helion_op5_op6_op7(buf3, buf4, arg3_1, buf6, buf7)
        del arg3_1
        del buf3
        del buf4
    return (buf6, buf7)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [512, 512], 'dtype': 'float32', 'stride': [1, 512], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8192, 512], 'dtype': 'bfloat16', 'stride': [512, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 512], 'dtype': 'bfloat16', 'stride': [1, 512], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [8, 8, 1024, 1024], 'dtype': 'float32', 'stride': [8388608, 1, 8192, 8], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [8, 1024, 8, 64]}, {'kind': 'shape', 'name': None, 'dims': [8, 1024, 8, 64]}, {'kind': 'shape', 'name': None, 'dims': [64, 1024, 64]}, {'kind': 'shape', 'name': None, 'dims': [64, 64, 1024]}, {'kind': 'shape', 'name': None, 'dims': [8, 8, 1024, 1024]}]}
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
