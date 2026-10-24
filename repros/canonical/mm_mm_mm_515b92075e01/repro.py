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
    def forward(self, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        view_default = torch.ops.aten.view.default(input_41, input_42);  input_41 = input_42 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_43);  view_default = input_43 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_44);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_45, input_44);  input_45 = input_44 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        sigmoid_default = torch.ops.aten.sigmoid.default(convert_element_type_default_1)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, sigmoid_default);  convert_element_type_default = None
        sub_tensor = torch.ops.aten.sub.Tensor(1, sigmoid_default, alpha = 1);  sigmoid_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, sub_tensor);  convert_element_type_default_1 = sub_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, 1, alpha = 1);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_2, input_46);  convert_element_type_default_2 = input_46 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_47);  input_47 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_48);  permute_default = input_48 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = torch.float32);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_49);  sum_dim_int_list = input_49 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_4, torch.bfloat16);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_50);  mm_default_1 = input_50 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float32);  convert_element_type_default_3 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_51);  input_51 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 240)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None)
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_52, torch.float32);  input_52 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_7, input_53, alpha = 1);  convert_element_type_default_7 = input_53 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_1, input_54);  sub_tensor_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_5);  mul_tensor_3 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True, dtype = None);  mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_5, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list_1, alpha = 1);  mul_tensor_4 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_7, alpha = 1);  sub_tensor_2 = mul_tensor_7 = None
        div_tensor = torch.ops.aten.div.Tensor(input_54, 240);  input_54 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_3);  div_tensor = sub_tensor_3 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, mul_tensor_5);  mul_tensor_5 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], False, dtype = None);  mul_tensor_9 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 1], False, dtype = None);  convert_element_type_default_4 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        return (convert_element_type_default_5, convert_element_type_default_6, sum_dim_int_list_3, sum_dim_int_list_4, convert_element_type_default_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
