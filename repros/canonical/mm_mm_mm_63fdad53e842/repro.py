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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59):
        mul_tensor = torch.ops.aten.mul.Tensor(input_45, 1024)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_45, [3], True, dtype = None);  input_45 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_46, input_47);  input_46 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_48, [3], True, dtype = None);  input_48 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, sum_dim_int_list_1);  mul_tensor_1 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, sum_dim_int_list, alpha = 1);  mul_tensor = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_2, alpha = 1);  sub_tensor = mul_tensor_2 = None
        div_tensor = torch.ops.aten.div.Tensor(input_47, 1024);  input_47 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_1);  div_tensor = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_49);  convert_element_type_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(convert_element_type_default_1, 0.8999999985098839);  convert_element_type_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default, div_tensor_1);  view_default = div_tensor_1 = None
        view_default_1 = torch.ops.aten.view.default(mul_tensor_4, input_51);  mul_tensor_4 = input_51 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_52);  input_52 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_53);  permute_default = input_53 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_54);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(input_55, input_54);  input_55 = input_54 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.7071067811865476)
        erf_default = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, convert_element_type_default_4)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, -0.5);  mul_tensor_7 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_8);  mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, mul_tensor_9);  convert_element_type_default_4 = mul_tensor_9 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_10, alpha = 1);  mul_tensor_6 = mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, add_tensor_1);  convert_element_type_default_3 = add_tensor_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_11, torch.bfloat16);  mul_tensor_11 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_5, input_56);  convert_element_type_default_5 = input_56 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, input_57);  input_57 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_58);  permute_default_1 = input_58 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True, dtype = torch.float32);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(sum_dim_int_list_2, input_59);  sum_dim_int_list_2 = input_59 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_5, torch.bfloat16);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_49);  mm_default_2 = input_49 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        return (convert_element_type_default_2, convert_element_type_default_6, convert_element_type_default_7, convert_element_type_default_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
