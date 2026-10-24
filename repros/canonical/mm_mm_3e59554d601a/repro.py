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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45):
        getitem = input_35[0]
        getitem_1 = input_35[1]
        getitem_2 = input_35[2];  input_35 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(getitem_2, [0, -4], 0);  getitem_2 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(getitem_1, [0, -4], 0);  getitem_1 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(getitem, [0, -4], 0);  getitem = None
        cat_default = torch.ops.aten.cat.default([constant_pad_nd_default_2, constant_pad_nd_default_1, constant_pad_nd_default], 0);  constant_pad_nd_default_2 = constant_pad_nd_default_1 = constant_pad_nd_default = None
        view_default = torch.ops.aten.view.default(cat_default, input_36);  cat_default = input_36 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 3, 0, 2, 4]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_37);  clone_default = input_37 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_38);  view_default_1 = input_38 = None
        mm_default = torch.ops.aten.mm.default(view_default_2, input_39);  input_39 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_1, input_40);  permute_default_1 = input_40 = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_41);  mm_default = input_41 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_42);  input_42 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 240)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_43, torch.float32);  input_43 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_2, input_44, alpha = 1);  convert_element_type_default_2 = input_44 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_45);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4, alpha = 1);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor = torch.ops.aten.div.Tensor(input_45, 240);  input_45 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], False, dtype = None);  mul_tensor_6 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [0, 1], False, dtype = None);  convert_element_type_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        return (convert_element_type_default_1, sum_dim_int_list_2, sum_dim_int_list_3, convert_element_type_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
