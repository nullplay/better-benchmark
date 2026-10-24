"""8-hop region from a captured real-world FX graph.

Exact structure and source provenance: meta.json and ../../matmul_manifest.json.
All retained input settings: shapes.json.
"""
import sys
from math import inf, nan
from pathlib import Path

import torch
import torch._inductor.inductor_prims
NoneType = type(None)
import torch as _fx_global_device
device = _fx_global_device.device
import torch.fx._pytree as fx_pytree
import torch.utils._pytree as pytree

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import default_shape_config, make_inputs_from_config, benchmark_repro

_repro_version = 3


class Repro(torch.nn.Module):
    def forward(self, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94):
        add_tensor = torch.ops.aten.add.Tensor(input_72, input_73, alpha = 1);  input_72 = input_73 = None
        view_default = torch.ops.aten.view.default(input_74, input_75);  input_74 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        addmm_default = torch.ops.aten.addmm.default(input_76, input_77, input_78, beta = 1, alpha = 1);  input_76 = input_77 = input_78 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_75);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mean_dim, 1e-05, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_79, mul_tensor);  input_79 = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_1, input_80);  convert_element_type_default_1 = input_80 = None
        permute_default = torch.ops.aten.permute.default(input_81, [1, 0]);  input_81 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_82, view_default_2, permute_default, beta = 1, alpha = 1);  input_82 = permute_default = None
        topk_default = torch.ops.aten.topk.default(addmm_default_1, 4, -1, True, True);  addmm_default_1 = None
        getitem = topk_default[0]
        getitem_1 = topk_default[1];  topk_default = None
        view_default_3 = torch.ops.aten.view.default(getitem_1, input_83);  getitem_1 = None
        sort_default = torch.ops.aten.sort.default(view_default_3, -1, False);  view_default_3 = None
        getitem_2 = sort_default[0]
        getitem_3 = sort_default[1];  sort_default = None
        full_default = torch.ops.aten.full.default(input_84, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar = torch.ops.aten.ge.Scalar(getitem_2, 32)
        unsqueeze_default = torch.ops.aten.unsqueeze.default(ge_scalar, -1);  ge_scalar = None
        full_default_1 = torch.ops.aten.full.default(input_84, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_84 = None
        div_tensor_mode = torch.ops.aten.div.Tensor_mode(getitem_3, 4, rounding_mode = 'floor')
        index_tensor = torch.ops.aten.index.Tensor(view_default_2, [div_tensor_mode]);  view_default_2 = div_tensor_mode = None
        where_self = torch.ops.aten.where.self(unsqueeze_default, full_default_1, index_tensor);  full_default_1 = index_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(getitem_2, torch.int32)
        histc_default = torch.ops.aten.histc.default(convert_element_type_default_2, 32, 0, 31);  convert_element_type_default_2 = None
        cumsum_default = torch.ops.aten.cumsum.default(histc_default, 0, dtype = torch.int32);  histc_default = None
        _grouped_mm_default = torch.ops.aten._grouped_mm.default(where_self, input_85, cumsum_default, None, None);  where_self = input_85 = None
        clamp_max_default = torch.ops.aten.clamp_max.default(getitem_2, 31);  getitem_2 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(input_86, [clamp_max_default]);  input_86 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(_grouped_mm_default, index_tensor_1, alpha = 1);  _grouped_mm_default = index_tensor_1 = None
        slice_tensor = torch.ops.aten.slice.Tensor(add_tensor_4, 1, 1, 9223372036854775807, 2)
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        clamp_min_default = torch.ops.aten.clamp_min.default(convert_element_type_default_3, -7.0);  convert_element_type_default_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(input_87, 1, alpha = 1);  input_87 = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(add_tensor_4, 1, 0, 9223372036854775807, 2);  add_tensor_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(slice_tensor_1, torch.float32);  slice_tensor_1 = None
        clamp_max_default_1 = torch.ops.aten.clamp_max.default(convert_element_type_default_4, 7.0);  convert_element_type_default_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_88, input_89);  input_88 = input_89 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_5, mul_tensor_2);  add_tensor_5 = mul_tensor_2 = None
        _grouped_mm_default_1 = torch.ops.aten._grouped_mm.default(mul_tensor_3, input_90, cumsum_default, None, None);  mul_tensor_3 = input_90 = cumsum_default = None
        index_tensor_2 = torch.ops.aten.index.Tensor(input_91, [clamp_max_default]);  input_91 = clamp_max_default = None
        add_tensor_6 = torch.ops.aten.add.Tensor(_grouped_mm_default_1, index_tensor_2, alpha = 1);  _grouped_mm_default_1 = index_tensor_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_5, [1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_5, amax_default, alpha = 1);  convert_element_type_default_5 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_6, input_83);  convert_element_type_default_6 = input_83 = None
        index_tensor_3 = torch.ops.aten.index.Tensor(view_default_4, [getitem_3]);  view_default_4 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(index_tensor_3, -1);  index_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_6, unsqueeze_default_1);  add_tensor_6 = unsqueeze_default_1 = None
        where_self_1 = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor_4);  unsqueeze_default = full_default = mul_tensor_4 = None
        empty_memory_format = torch.ops.aten.empty.memory_format(input_92, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False, memory_format = None);  input_92 = None
        iota_default = torch.ops.prims.iota.default(4000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put_default = torch.ops.aten.index_put.default(empty_memory_format, [getitem_3], iota_default, False);  empty_memory_format = getitem_3 = iota_default = None
        index_tensor_4 = torch.ops.aten.index.Tensor(where_self_1, [index_put_default]);  where_self_1 = index_put_default = None
        view_default_5 = torch.ops.aten.view.default(index_tensor_4, input_93);  index_tensor_4 = input_93 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_5, [1], False, dtype = None);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list_1, input_75);  sum_dim_int_list_1 = input_75 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_6, alpha = 1);  add_tensor_2 = view_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.float32)
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_7, input_94, alpha = 1);  add_tensor_7 = input_94 = None
        return (clamp_min_default, clamp_max_default_1, convert_element_type_default_7, add_tensor_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
