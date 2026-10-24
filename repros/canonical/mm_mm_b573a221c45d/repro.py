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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49):
        mul_tensor = torch.ops.aten.mul.Tensor(input_35, 512)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_35, [2], True, dtype = None);  input_35 = None
        view_default = torch.ops.aten.view.default(input_36, input_37);  input_36 = None
        relu_default = torch.ops.aten.relu.default(view_default);  view_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(relu_default, torch.float32)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_38, alpha = 1);  convert_element_type_default = input_38 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, input_39);  sub_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_40, [2], True, dtype = None);  input_40 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, sum_dim_int_list_1);  mul_tensor_1 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor, sum_dim_int_list, alpha = 1);  mul_tensor = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        div_tensor = torch.ops.aten.div.Tensor(input_39, 512);  input_39 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default = torch.ops.aten.full.default(input_41, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_41 = None
        where_self = torch.ops.aten.where.self(le_scalar, full_default, convert_element_type_default_1);  le_scalar = full_default = convert_element_type_default_1 = None
        view_default_1 = torch.ops.aten.view.default(where_self, input_42);  where_self = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_43);  input_43 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_44);  permute_default = input_44 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_37);  mm_default = input_37 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 1], True, dtype = torch.float32)
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list_2, input_45);  sum_dim_int_list_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_46);  input_46 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_47);  convert_element_type_default_2 = input_47 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True, dtype = torch.float32);  mul_tensor_4 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_3, input_45);  sum_dim_int_list_3 = input_45 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16)
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_4, input_42);  convert_element_type_default_4 = input_42 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_48, torch.float32);  input_48 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_5, convert_element_type_default_5, alpha = 1);  mul_tensor_5 = convert_element_type_default_5 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, input_49, alpha = 1);  add_tensor = input_49 = None
        return (convert_element_type_default_3, view_default_3, view_default_4, view_default_5, add_tensor_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
