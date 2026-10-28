"""First-shape Helion native MM wrapper; generated without execution.
Repro: bmm_d4a8bd8d80de
Shape hash: 943778ac
call(args) follows the Inductor boxed calling convention.
"""
import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision='ieee')
def helion_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_z, tile_y, tile_x in hl.tile([512, 128, 128]):
        acc = hl.zeros([tile_z, tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(64):
            tmp0 = A[tile_z, tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_z, tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.baddbmm(acc, tmp1, tmp3)
        out0[tile_z, tile_y, tile_x] = acc.to(torch.bfloat16)

def call_helion_op2(arg0_1: torch.Tensor, arg1_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op2(arg0_1, arg1_1, buf2)
import torch
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._C._dynamo.guards import copy_if_misaligned
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
aten = torch.ops.aten
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

@triton_heuristics.persistent_reduction(size_hints={'x': 65536, 'r0_': 128}, reduction_hint=ReductionHint.INNER, filename=os.path.join(cache_dir(), 'cv', 'ccvibreldnks77dwvgiytw3dv2mfly3eoqvgkiii2wssxghhuqjf.py'), triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'load_seed_offset': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=132, cc=90, major=9, regs_per_multiprocessor=65536, max_threads_per_multi_processor=2048, max_threads_per_block=1024, warp_size=32), 'constants': {}, 'native_matmul': False, 'enable_fp_fusion': True, 'launch_pdl': False, 'disable_ftz': False, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]}, inductor_meta={'grid_type': 'Grid1D', 'kernel_name': 'triton_per_fused_add_convert_element_type_div_exp_full_gt_inductor_lookup_seed_inductor_random_maximum_mul_prepare_softmax_online_sub_view_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': None, 'atomic_add_found': False, 'num_load': 2, 'num_store': 3, 'num_reduction': 4, 'autotune_hints': (), 'tiling_scores': {'x': 1048576, 'r0_': 52428800}, 'backend_hash': '36D91BE546B0C7EE6918BC42CC20690A0C8FA2E2A4F2245039508CE26055BE6D', 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'incremental_autotune': False, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'batch_invariant': False, 'force_filter_reduction_configs': False, 'mix_order_reduction_allow_multi_stages': True, 'dynamic_disable_pipelining': True, 'are_deterministic_algorithms_enabled': False})
@triton.jit
def triton_per_fused_add_convert_element_type_div_exp_full_gt_inductor_lookup_seed_inductor_random_maximum_mul_prepare_softmax_online_sub_view_1(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, load_seed_offset, xnumel, r0_numel, XBLOCK: tl.constexpr):
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
    x1 = xindex // 128
    tmp0 = tl.load(in_out_ptr0 + (r0_2 + 128 * x3), None, eviction_policy='evict_first').to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (r0_2 + 128 * x0 + 16384 * (x1 // 16)), None, eviction_policy='evict_first')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.full([1, 1], -3.4028234663852886e+38, tl.float32)
    tmp5 = tl.maximum(tmp3, tmp4, tl.PropagateNan.ALL)
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp10 = triton_helpers.fmax2(tmp8, 1)[:, None].to(tl.float32)
    tmp11 = tmp6 - tmp10
    tmp12 = libdevice.exp(tmp11)
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = tl.load(in_ptr1 + load_seed_offset)
    tmp17 = (r0_2 + 128 * x3).to(tl.int32)
    tmp18 = tl.rand(tmp16, tmp17.to(tl.uint32))
    tmp19 = tl.full([1, 1], 0.1, tl.float32)
    tmp20 = tmp18 > tmp19
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp5 - tmp10
    tmp23 = libdevice.exp(tmp22)
    tmp24 = tmp23 / tmp15
    tmp25 = tmp21 * tmp24
    tmp26 = tl.full([1, 1], 1.1111111111111112, tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp27.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_2 + 128 * x3), tmp28, None)
    tl.store(out_ptr0 + x3, tmp10, None)
    tl.store(out_ptr1 + x3, tmp15, None)
async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((3,), (1,), torch.int64)
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [3], out=buf0)
        assert_size_stride_grouped((arg0_1, arg1_1), ((512, 128, 64), (512, 64, 128)), ((8192, 64, 1), (8192, 1, 64)), 'input')
        arg0_1 = copy_if_misaligned(arg0_1)
        arg1_1 = copy_if_misaligned(arg1_1)
        buf2 = empty_strided_cuda((512, 128, 128), (16384, 128, 1), torch.bfloat16)
        raw_stream0 = get_raw_stream(0)
        call_helion_op2(arg0_1, arg1_1, buf2)
        del arg0_1
        del arg1_1
        assert_size_stride(arg2_1, (32, 1, 128, 128), (16384, 16384, 128, 1), 'input')
        arg2_1 = copy_if_misaligned(arg2_1)
        buf3 = empty_strided_cuda((512, 128, 1), (128, 1, 1), torch.float32)
        buf4 = empty_strided_cuda((512, 128, 1), (128, 1, 1), torch.float32)
        buf5 = buf2
        del buf2
        raw_stream0 = get_raw_stream(0)
        triton_per_fused_add_convert_element_type_div_exp_full_gt_inductor_lookup_seed_inductor_random_maximum_mul_prepare_softmax_online_sub_view_1.run(buf5, arg2_1, buf0, buf3, buf4, 0, 65536, 128, stream=raw_stream0)
        del arg2_1
        del buf0
    return (buf3, buf4, buf5, reinterpret_tensor(buf5, (512, 128, 128), (16384, 1, 128), 0))

def benchmark_compiled_module(args, times=10, repeat=10):
    from torch._inductor.utils import print_performance
    fn = lambda: call(list(args))
    return print_performance(fn, times=times, repeat=repeat, device='cuda')
FIRST_SHAPE_CONFIG = {'inputs': [{'kind': 'tensor', 'name': None, 'shape': [512, 128, 64], 'dtype': 'bfloat16', 'stride': [8192, 64, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [512, 64, 128], 'dtype': 'bfloat16', 'stride': [8192, 1, 64], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'tensor', 'name': None, 'shape': [32, 1, 128, 128], 'dtype': 'float32', 'stride': [16384, 16384, 128, 1], 'device': 'cuda', 'storage_offset': 0}, {'kind': 'shape', 'name': None, 'dims': [32, 16, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [512, 128, 128]}, {'kind': 'shape', 'name': None, 'dims': [512, 128, 128]}]}
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
