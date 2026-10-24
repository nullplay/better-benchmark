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
    def forward(self, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_47, torch.bfloat16)
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_48, torch.bfloat16);  input_48 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        view_default = torch.ops.aten.view.default(mul_tensor_1, input_49);  mul_tensor_1 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_50);  input_50 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_51);  permute_default = input_51 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_52);  mm_default = input_52 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        full_default = torch.ops.aten.full.default(input_53, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_53 = None
        where_self = torch.ops.aten.where.self(input_54, full_default, view_default_1);  input_54 = full_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(where_self, input_55);  where_self = input_55 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_56);  input_56 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_57);  permute_default_1 = input_57 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True, dtype = torch.float32);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list, input_58);  sum_dim_int_list = input_58 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_3, torch.bfloat16);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_59);  mm_default_2 = input_59 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float32);  convert_element_type_default_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_60);  input_60 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1024)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_61);  mul_tensor_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_61, input_62);  input_62 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_1, alpha = 1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_5, alpha = 1);  sub_tensor = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_63, sub_tensor_1);  input_63 = sub_tensor_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_61);  input_61 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], False, dtype = None);  mul_tensor_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 1], False, dtype = None);  convert_element_type_default_4 = None
        add_tensor = torch.ops.aten.add.Tensor(input_47, mul_tensor_6, alpha = 1);  input_47 = mul_tensor_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_64, 1.1111111111111112);  input_64 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, mul_tensor_8);  convert_element_type_default_7 = mul_tensor_8 = None
        view_default_5 = torch.ops.aten.view.default(mul_tensor_9, input_49);  mul_tensor_9 = input_49 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(input_65, input_66, alpha = 1);  input_65 = input_66 = None
        div_tensor = torch.ops.aten.div.Tensor(input_67, 1024);  input_67 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_10, alpha = 1);  add_tensor = mul_tensor_10 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16)
        mul_tensor_11 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, input_68);  convert_element_type_default_8 = input_68 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_69, input_70);  input_69 = input_70 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_12, alpha = 1);  add_tensor_1 = mul_tensor_12 = None
        return (convert_element_type_default_2, convert_element_type_default_5, convert_element_type_default_6, mul_tensor_4, sum_dim_int_list_2, sum_dim_int_list_3, view_default_5, mul_tensor_11, add_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
