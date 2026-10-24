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
    def forward(self, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61):
        add_tensor = torch.ops.aten.add.Tensor(input_38, input_39, alpha = 1);  input_38 = input_39 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_40, input_41);  input_40 = input_41 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_42, input_43, alpha = 1);  input_42 = input_43 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_44, sub_tensor);  input_44 = sub_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_45, input_46);  input_45 = input_46 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_47, input_48, alpha = 1);  input_47 = input_48 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_49, sub_tensor_2);  input_49 = sub_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_50, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_50, [2], True, dtype = None);  input_50 = None
        view_default = torch.ops.aten.view.default(input_51, input_52);  input_51 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_53, view_default);  view_default = None
        add_tensor_4 = torch.ops.aten.add.Tensor(input_54, mul_tensor_5, alpha = 1);  input_54 = mul_tensor_5 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_4, input_55, alpha = 1);  add_tensor_4 = input_55 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_56);  sub_tensor_3 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_57, [2], True, dtype = None);  input_57 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, sum_dim_int_list_1);  mul_tensor_6 = sum_dim_int_list_1 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_7, alpha = 1);  sub_tensor_4 = mul_tensor_7 = None
        div_tensor = torch.ops.aten.div.Tensor(input_56, 768);  input_56 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_5);  div_tensor = sub_tensor_5 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_8, alpha = 1);  add_tensor_3 = mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor_5, input_53);  add_tensor_5 = input_53 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default, input_58);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_59);  input_59 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(input_60, [0, 2, 1, 3]);  input_60 = None
        view_default_2 = torch.ops.aten.view.default(permute_default_1, input_52);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_58);  view_default_2 = input_58 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, view_default_3);  permute_default = view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mm_default, input_52);  mm_default = input_52 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_61);  view_default_4 = input_61 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        return (convert_element_type_default_1, permute_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
