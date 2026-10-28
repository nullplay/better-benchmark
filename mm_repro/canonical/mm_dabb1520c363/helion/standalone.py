"""First-shape Helion native MM wrapper; generated without execution.
Repro: mm_dabb1520c363
Shape hash: d102a86e
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl
from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op4(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([16384, 1024]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(1024):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op4(buf2: torch.Tensor, buf3: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op4(reinterpret_tensor(buf2, (16384, 1024), (1024, 1), 0), reinterpret_tensor(buf3, (1024, 1024), (1024, 1), 0), buf4)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
aten = torch.ops.aten
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

@triton_heuristics.pointwise(size_hints={'x': 16777216}, filename=os.path.join(cache_dir(), 'jy', 'cjyohcizbi5jnk5a3og4fqt2g25irqd2buon2zhqkwahgjqtjwwh.py'), triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr1': '*bf16', 'load_seed_offset': 'constexpr', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {'load_seed_offset': 1}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_cat_convert_element_type_cos_div_expand_gt_inductor_lookup_seed_inductor_random_iota_mul_permute_pow_sin_unsqueeze_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 0, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 67108864}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_cat_convert_element_type_cos_div_expand_gt_inductor_lookup_seed_inductor_random_iota_mul_permute_pow_sin_unsqueeze_0(in_ptr0, out_ptr1, load_seed_offset, xnumel, XBLOCK: tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = xindex
    x1 = xindex % 1024
    x3 = xindex // 16384
    tmp0 = tl.load(in_ptr0 + load_seed_offset)
    tmp1 = x0.to(tl.int32)
    tmp2 = tl.rand(tmp0, tmp1.to(tl.uint32))
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tl.full([1], 0.10009765625, tl.float32)
    tmp5 = tmp3 > tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = x1.to(tl.int32)
    tmp8 = tl.full([1], 0, tl.int64)
    tmp9 = tmp7 >= tmp8
    tmp10 = x1.to(tl.int64)
    tmp11 = tmp10.to(tl.int64)
    tmp12 = tl.full([1], 512, tl.int64)
    tmp13 = tmp11 < tmp12
    tmp14 = (2 * x1).to(tl.float32)
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tl.full([1], 0.0009765625, tl.float32)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full([1], 10000.0, tl.float32)
    tmp19 = libdevice.pow(tmp18, tmp17)
    tmp20 = tl.full([1], 1.0, tl.float32)
    tmp21 = tmp20 / tmp19
    tmp22 = tmp21.to(tl.float32)
    tmp23 = (512 + -1 * x3).to(tl.float32)
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 * tmp22
    tmp26 = tl_math.sin(tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp13, tmp26, tmp27)
    tmp29 = tmp7 >= tmp12
    tmp30 = tl.full([1], 1024, tl.int64)
    tmp31 = tmp7 < tmp30
    tmp32 = (2 * (-512 + x1)).to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tl.full([1], 0.0009765625, tl.float32)
    tmp35 = tmp33 * tmp34
    tmp36 = tl.full([1], 10000.0, tl.float32)
    tmp37 = libdevice.pow(tmp36, tmp35)
    tmp38 = tl.full([1], 1.0, tl.float32)
    tmp39 = tmp38 / tmp37
    tmp40 = tmp39.to(tl.float32)
    tmp41 = (512 + -1 * x3).to(tl.float32)
    tmp42 = tmp41.to(tl.float32)
    tmp43 = tmp42 * tmp40
    tmp44 = tl_math.cos(tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp29, tmp44, tmp45)
    tmp47 = tl.where(tmp13, tmp28, tmp46)
    tmp48 = tmp6 * tmp47
    tmp49 = tl.full([1], 1.1111111111111112, tl.float32)
    tmp50 = tmp48 * tmp49
    tl.store(out_ptr1 + x0, tmp50, None)
import os
from torch._inductor.runtime.runtime_utils import cache_dir
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(size_hints={'x': 1048576}, filename=os.path.join(cache_dir(), 'ha', 'cha56slbdcqgncefcx5doxbrvw7gtluuvhvn6x43m4t5tdt2movf.py'), triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_poi_fused_convert_element_type_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'atomic_add_found': False, 'num_load': 1, 'num_store': 1, 'num_reduction': 0, 'autotune_hints': (), 'tiling_scores': {'x': 8388608}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False}, min_elem_per_thread=0)
@triton.jit
def triton_poi_fused_convert_element_type_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xnumel = 1048576
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
    arg0_1, = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((99,), (1,), torch.int64)
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [99], out=buf0)
        buf2 = empty_strided_cuda((1024, 16, 1024), (16384, 1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convert_element_type_cos_div_expand_gt_inductor_lookup_seed_inductor_random_iota_mul_permute_pow_sin_unsqueeze_0.run(buf0, buf2, 1, 16777216, stream=raw_stream0)
        del buf0
        assert_size_stride(arg0_1, (1024, 16, 64), (1024, 64, 1), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        buf3 = empty_strided_cuda((1024, 16, 64), (1024, 64, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        triton_poi_fused_convert_element_type_1.run(arg0_1, buf3, 1048576, stream=raw_stream0)
        del arg0_1
        buf4 = empty_strided_cuda((16384, 1024), (1024, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op4(buf2, buf3, buf4)
        del buf2
        del buf3
    return (reinterpret_tensor(buf4, (256, 64, 1024), (64, 1, 16384), 0), reinterpret_tensor(buf4, (256, 1024, 64), (64, 16384, 1), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [1024, 16, 64], 'dtype': 'float32', 'stride': [1024, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [-1, 16, -1]}, {'kind': 'shape', 'name': None, 'dims': [1024, 16, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1, 16384, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1, 1024, 1024]}, {'kind': 'shape', 'name': None, 'dims': [1024, 16, 1, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [1024, 16, 16, 64]}, {'kind': 'shape', 'name': None, 'dims': [256, 64, 1024]}]}
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
