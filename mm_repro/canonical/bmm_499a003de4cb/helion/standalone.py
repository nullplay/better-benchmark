"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_499a003de4cb
Shape hash: 5246f0df
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([256, 64, 1024]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[(tile_x.index.reshape([1, 1, tile_x]) + 1024 * tile_y.index.reshape([1, tile_y, 1]) + 65536 * tile_z.index.reshape([tile_z, 1, 1])) % 1024, tile_z.index.reshape([tile_z, 1, 1]) // 16, 0, tile_z.index.reshape([tile_z, 1, 1]) % 16, tile_y.index.reshape([1, tile_y, 1])] = acc.to(torch.bfloat16)

def call_helion_op1_op2(arg1_1: torch.Tensor, buf0: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op1_op2(arg1_1, reinterpret_tensor(buf0, (256, 512, 1024), (524288, 1024, 1), 0), buf2)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
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

@triton_heuristics.pointwise(size_hints={'x': 134217728}, filename=os.path.join(cache_dir(), 'ra', 'cral42okhk4hvvw23vizbqhsgd6loakl5nhoh5frx5bifbefr5kj.py'), triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_full_view_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 805044224}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_full_view_0(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 134217728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x1 = xindex // 512 % 1024
    x2 = xindex // 524288
    x3 = xindex % 524288
    x4 = xindex
    tmp0 = x1.to(tl.int64)
    tmp1 = tmp0.to(tl.int64)
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.load(in_ptr0 + (-512 + x3 + 523776 * x2), tmp3, other=0.0).to(tl.float32)
    tmp5 = tl.full([1], 0.0, tl.float32)
    tmp6 = tl.where(tmp3, tmp4, tmp5)
    tl.store(out_ptr0 + x4, tmp6, None)
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
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 16, 512, 1023), (8380416, 523776, 1023, 1), 'input')
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        arg0_1 = copy_if_misaligned(arg0_1)
        buf0 = empty_strided_cuda((16, 16, 1024, 512), (8388608, 524288, 512, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_full_view_0.run(arg0_1, buf0, 134217728, stream=raw_stream0)
        del arg0_1
        assert_size_stride(arg1_1, (256, 64, 512), (64, 1, 16384), 'input')
        arg1_1 = copy_if_misaligned(arg1_1)
        buf2 = empty_strided_cuda((1024, 16, 1, 16, 64), (16384, 1024, 1024, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1_op2(arg1_1, buf0, buf2)
        del arg1_1
        del buf0
    return (reinterpret_tensor(buf2, (16384, 1024), (1024, 1), 0),)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [16, 16, 512, 1023], 'dtype': 'bfloat16', 'stride': [8380416, 523776, 1023, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [256, 64, 512], 'dtype': 'bfloat16', 'stride': [64, 1, 16384], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [16, 16, 1023, 512]}, {'kind': 'shape', 'name': None, 'dims': [16, 16, 1024, 512]}, {'kind': 'shape', 'name': None, 'dims': [16, 16, 512, 1024]}, {'kind': 'shape', 'name': None, 'dims': [16, 16, 512, 1024, 1]}, {'kind': 'shape', 'name': None, 'dims': [256, 512, 1024]}, {'kind': 'shape', 'name': None, 'dims': [16, 16, 64, 1024, 1]}, {'kind': 'shape', 'name': None, 'dims': [1024, 16, 16, 64, 1]}, {'kind': 'shape', 'name': None, 'dims': [1, 16384, 1024]}]}
COMPILED_INPUT_ORDER = [0, 1]

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
