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
    def forward(self, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68):
        mul_tensor = torch.ops.aten.mul.Tensor(input_48, input_49);  input_48 = input_49 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_50, input_51, alpha = 1);  input_50 = input_51 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor, alpha = 1);  sub_tensor = mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_52, sub_tensor_1);  input_52 = sub_tensor_1 = None
        view_default = torch.ops.aten.view.default(input_53, input_54);  input_53 = None
        as_strided_default = torch.ops.aten.as_strided.default(input_55, input_56, input_57, input_58);  input_55 = input_56 = input_57 = input_58 = None
        view_default_1 = torch.ops.aten.view.default(as_strided_default, input_59);  as_strided_default = input_59 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_60);  view_default_1 = input_60 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [1, 0, 2, 3]);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(permute_default_1, input_54);  permute_default_1 = None
        div_tensor = torch.ops.aten.div.Tensor(view_default_3, 8.0);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(view_default, input_61);  view_default = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default_2, input_62);  permute_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_4, input_63);  view_default_4 = input_63 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_54);  mm_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        view_default_6 = torch.ops.aten.view.default(input_64, input_54);  input_64 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_61);  view_default_6 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default_3, input_62);  permute_default_3 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_7, input_65);  view_default_7 = input_65 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_3, input_54);  mm_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_2, alpha = 1);  convert_element_type_default = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        view_default_9 = torch.ops.aten.view.default(div_tensor, input_61);  div_tensor = input_61 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_9, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_4, input_62);  permute_default_4 = input_62 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_9, input_66);  view_default_9 = input_66 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_5, input_54);  mm_default_5 = input_54 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_4, alpha = 1);  add_tensor = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        permute_default_5 = torch.ops.aten.permute.default(add_tensor_1, [1, 0, 2]);  add_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, permute_default_5, alpha = 1);  mul_tensor_1 = permute_default_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_2, input_67);  input_67 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_68);  mul_tensor_2 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True, dtype = None);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_68, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list, alpha = 1);  mul_tensor_3 = sum_dim_int_list = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_2, input_68);  input_68 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], False, dtype = None);  mul_tensor_6 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], False, dtype = None);  add_tensor_2 = None
        return (convert_element_type_default_1, convert_element_type_default_3, convert_element_type_default_5, mul_tensor_5, sub_tensor_2, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
