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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        getitem = input_35[0]
        getitem_1 = input_35[1]
        getitem_2 = input_35[2];  input_35 = None
        permute_default = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default = torch.ops.aten.view.default(permute_default, input_36);  permute_default = None
        permute_default_1 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        view_default_1 = torch.ops.aten.view.default(permute_default_1, input_36);  permute_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(getitem_1, [0, 2, 1, 3]);  getitem_1 = None
        view_default_2 = torch.ops.aten.view.default(permute_default_2, input_36);  permute_default_2 = None
        cat_default = torch.ops.aten.cat.default([view_default, view_default_2, view_default_1], 2);  view_default = view_default_2 = view_default_1 = None
        view_default_3 = torch.ops.aten.view.default(cat_default, input_37);  cat_default = input_37 = None
        mm_default = torch.ops.aten.mm.default(view_default_3, input_38);  input_38 = None
        mm_default_1 = torch.ops.aten.mm.default(input_39, view_default_3);  input_39 = view_default_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default, input_36);  convert_element_type_default = input_36 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_4, input_40);  input_40 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        add_tensor = torch.ops.aten.add.Tensor(input_41, input_42, alpha = 1);  input_41 = input_42 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_43, add_tensor);  input_43 = add_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_3, input_44, alpha = 1);  mul_tensor_3 = input_44 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor, input_45);  sub_tensor = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_4);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None);  mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_6, alpha = 1);  sub_tensor_1 = mul_tensor_6 = None
        div_tensor = torch.ops.aten.div.Tensor(input_45, 768);  input_45 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_default_4, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1], False, dtype = None);  mul_tensor_8 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_4, [0, 1], False, dtype = None);  view_default_4 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_46, mul_tensor_7, alpha = 1);  input_46 = mul_tensor_7 = None
        return (convert_element_type_default_1, sum_dim_int_list_2, sum_dim_int_list_3, add_tensor_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
