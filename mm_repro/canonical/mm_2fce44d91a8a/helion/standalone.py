"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_2fce44d91a8a
Shape hash: a1c20a52
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op1(buf0: torch.Tensor, arg0_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op1(reinterpret_tensor(buf0, (16384, 1024), (1024, 1), 0), reinterpret_tensor(arg0_1, (1024, 1024), (1024, 1), 0), buf1)
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

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), 'ak', 'caklfhwzksigh7brumihchssw7hck5cwpa7wjkjeloufggkk7vol.py'), triton_meta={'signature': {'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_cat_clone_convert_element_type_cos_div_expand_iota_mul_permute_pow_sin_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 67108864}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_cat_clone_convert_element_type_cos_div_expand_iota_mul_permute_pow_sin_unsqueeze_0(out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex % 1024
    x2 = xindex // 16384
    x5 = xindex
    tmp0 = x0.to(tl.int32)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = x0.to(tl.int64)
    tmp4 = tmp3.to(tl.int64)
    tmp5 = tl.full([1], 512, tl.int64)
    tmp6 = tmp4 < tmp5
    tmp7 = (2 * x0).to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full([1], 0.0009765625, tl.float32)
    tmp10 = tmp8 * tmp9
    tmp11 = tl.full([1], 10000.0, tl.float32)
    tmp12 = libdevice.pow(tmp11, tmp10)
    tmp13 = tl.full([1], 1.0, tl.float32)
    tmp14 = tmp13 / tmp12
    tmp15 = (512 + -1 * x2).to(tl.float32)
    tmp16 = tmp15.to(tl.float32)
    tmp17 = tmp16 * tmp14
    tmp18 = tl_math.sin(tmp17)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp6, tmp18, tmp19)
    tmp21 = tmp0 >= tmp5
    tmp22 = tl.full([1], 1024, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = (2 * (-512 + x0)).to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tl.full([1], 0.0009765625, tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tl.full([1], 10000.0, tl.float32)
    tmp29 = libdevice.pow(tmp28, tmp27)
    tmp30 = tl.full([1], 1.0, tl.float32)
    tmp31 = tmp30 / tmp29
    tmp32 = (512 + -1 * x2).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp33 * tmp31
    tmp35 = tl_math.cos(tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp21, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp20, tmp37)
    tmp39 = tmp38.to(tl.float32)
    tl.store(out_ptr0 + x5, tmp39, None)
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
    arg0_1, = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1024, 16, 1024), (16384, 1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_cat_clone_convert_element_type_cos_div_expand_iota_mul_permute_pow_sin_unsqueeze_0.run(buf0, 16777216, stream=raw_stream0)
        assert_size_stride(arg0_1, (1024, 16, 64), (1024, 64, 1), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf1 = empty_strided_cuda((16384, 1024), (1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op1(buf0, arg0_1, buf1)
        del arg0_1
        del buf0
    return (reinterpret_tensor(buf1, (256, 64, 1024), (64, 1, 16384), 0),)

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1024, 16, 64], 'dtype': 'bfloat16', 'stride': [1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [-1, 16, -1]}, {'kind': 'shape', 'name': None, 'dims': [1, 16384, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1, 1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1024, 16, 1, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [1024, 16, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [256, 64, 1024]}]}
COMPILED_INPUT_ORDER = [0]

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
