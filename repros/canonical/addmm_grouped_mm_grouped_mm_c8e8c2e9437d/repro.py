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
    def forward(self, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77):
        mul_tensor = torch.ops.aten.mul.Tensor(input_61, input_62);  input_61 = input_62 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_63, mul_tensor);  input_63 = mul_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_64);  convert_element_type_default = input_64 = None
        permute_default = torch.ops.aten.permute.default(input_65, [1, 0]);  input_65 = None
        addmm_default = torch.ops.aten.addmm.default(input_66, view_default, permute_default, beta = 1, alpha = 1);  input_66 = permute_default = None
        topk_default = torch.ops.aten.topk.default(addmm_default, 4, -1, True, True);  addmm_default = None
        getitem = topk_default[0]
        getitem_1 = topk_default[1];  topk_default = None
        view_default_1 = torch.ops.aten.view.default(getitem_1, input_67);  getitem_1 = None
        sort_default = torch.ops.aten.sort.default(view_default_1, -1, False);  view_default_1 = None
        getitem_2 = sort_default[0]
        getitem_3 = sort_default[1];  sort_default = None
        full_default = torch.ops.aten.full.default(input_68, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge_scalar = torch.ops.aten.ge.Scalar(getitem_2, 32)
        unsqueeze_default = torch.ops.aten.unsqueeze.default(ge_scalar, -1);  ge_scalar = None
        full_default_1 = torch.ops.aten.full.default(input_68, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_68 = None
        div_tensor_mode = torch.ops.aten.div.Tensor_mode(getitem_3, 4, rounding_mode = 'floor')
        index_tensor = torch.ops.aten.index.Tensor(view_default, [div_tensor_mode]);  view_default = div_tensor_mode = None
        where_self = torch.ops.aten.where.self(unsqueeze_default, full_default_1, index_tensor);  full_default_1 = index_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(getitem_2, torch.int32)
        histc_default = torch.ops.aten.histc.default(convert_element_type_default_1, 32, 0, 31);  convert_element_type_default_1 = None
        cumsum_default = torch.ops.aten.cumsum.default(histc_default, 0, dtype = torch.int32);  histc_default = None
        _grouped_mm_default = torch.ops.aten._grouped_mm.default(where_self, input_69, cumsum_default, None, None);  where_self = input_69 = None
        clamp_max_default = torch.ops.aten.clamp_max.default(getitem_2, 31);  getitem_2 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(input_70, [clamp_max_default]);  input_70 = None
        add_tensor = torch.ops.aten.add.Tensor(_grouped_mm_default, index_tensor_1, alpha = 1);  _grouped_mm_default = index_tensor_1 = None
        slice_tensor = torch.ops.aten.slice.Tensor(add_tensor, 1, 1, 9223372036854775807, 2)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        clamp_min_default = torch.ops.aten.clamp_min.default(convert_element_type_default_2, -7.0);  convert_element_type_default_2 = None
        clamp_max_default_1 = torch.ops.aten.clamp_max.default(clamp_min_default, 7.0);  clamp_min_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(clamp_max_default_1, torch.bfloat16);  clamp_max_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_3, 1, alpha = 1);  convert_element_type_default_3 = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(add_tensor, 1, 0, 9223372036854775807, 2);  add_tensor = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(slice_tensor_1, torch.float32);  slice_tensor_1 = None
        clamp_max_default_2 = torch.ops.aten.clamp_max.default(convert_element_type_default_4, 7.0);  convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(clamp_max_default_2, torch.bfloat16);  clamp_max_default_2 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1.702)
        sigmoid_default = torch.ops.aten.sigmoid.default(mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, sigmoid_default);  convert_element_type_default_5 = sigmoid_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_3);  add_tensor_1 = mul_tensor_3 = None
        _grouped_mm_default_1 = torch.ops.aten._grouped_mm.default(mul_tensor_4, input_71, cumsum_default, None, None);  mul_tensor_4 = input_71 = cumsum_default = None
        index_tensor_2 = torch.ops.aten.index.Tensor(input_72, [clamp_max_default]);  input_72 = clamp_max_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(_grouped_mm_default_1, index_tensor_2, alpha = 1);  _grouped_mm_default_1 = index_tensor_2 = None
        exp_default = torch.ops.aten.exp.default(input_73);  input_73 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_6, input_67);  convert_element_type_default_6 = input_67 = None
        index_tensor_3 = torch.ops.aten.index.Tensor(view_default_2, [getitem_3]);  view_default_2 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(index_tensor_3, -1);  index_tensor_3 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_2, unsqueeze_default_1);  add_tensor_2 = unsqueeze_default_1 = None
        where_self_1 = torch.ops.aten.where.self(unsqueeze_default, full_default, mul_tensor_5);  unsqueeze_default = full_default = mul_tensor_5 = None
        empty_memory_format = torch.ops.aten.empty.memory_format(input_74, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False, memory_format = None);  input_74 = None
        iota_default = torch.ops.prims.iota.default(4000, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_put_default = torch.ops.aten.index_put.default(empty_memory_format, [getitem_3], iota_default, False);  empty_memory_format = getitem_3 = iota_default = None
        index_tensor_4 = torch.ops.aten.index.Tensor(where_self_1, [index_put_default]);  where_self_1 = index_put_default = None
        view_default_3 = torch.ops.aten.view.default(index_tensor_4, input_75);  index_tensor_4 = input_75 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_3, [1], False, dtype = None);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_1, input_76);  sum_dim_int_list_1 = input_76 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_77, view_default_4, alpha = 1);  input_77 = view_default_4 = None
        return (getitem, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
