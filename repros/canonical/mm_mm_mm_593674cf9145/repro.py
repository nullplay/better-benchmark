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
    def forward(self, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83):
        mul_tensor = torch.ops.aten.mul.Tensor(input_56, input_57);  input_56 = input_57 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 192)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(input_58, input_59, alpha = 1);  input_58 = input_59 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_60);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4, alpha = 1);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor = torch.ops.aten.div.Tensor(input_60, 192);  input_60 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_61)
        mm_default = torch.ops.aten.mm.default(view_default, input_62);  input_62 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_63);  permute_default = input_63 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_64);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_65, torch.float32);  input_65 = None
        sigmoid_default = torch.ops.aten.sigmoid.default(convert_element_type_default_3);  convert_element_type_default_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, sigmoid_default);  convert_element_type_default_2 = sigmoid_default = None
        add_tensor = torch.ops.aten.add.Tensor(input_66, 1, alpha = 1);  input_66 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, add_tensor);  mul_tensor_6 = add_tensor = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_67, input_68);  input_67 = input_68 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(input_69, input_70, alpha = 1);  input_69 = input_70 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_8, alpha = 1);  sub_tensor_3 = mul_tensor_8 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(input_71, 192);  input_71 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_4);  div_tensor_1 = sub_tensor_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_5, alpha = 1);  convert_element_type_default = convert_element_type_default_5 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_61)
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_72);  input_72 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(input_73, input_61);  input_73 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, view_default_3);  permute_default_1 = view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_74);  mm_default_2 = input_74 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_75);  view_default_4 = input_75 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(input_76, input_77, alpha = 1);  input_76 = input_77 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(input_78, 192);  input_78 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(div_tensor_2, sub_tensor_5);  div_tensor_2 = sub_tensor_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_7, alpha = 1);  add_tensor_1 = convert_element_type_default_7 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_2, input_61)
        mm_default_4 = torch.ops.aten.mm.default(view_default_6, input_79);  input_79 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [1, 0]);  view_default_6 = None
        mm_default_5 = torch.ops.aten.mm.default(permute_default_2, input_80);  permute_default_2 = input_80 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_4, input_64);  mm_default_4 = input_64 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_81, input_82);  input_81 = input_82 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mul_tensor_11, torch.bfloat16);  mul_tensor_11 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_8, alpha = 1);  add_tensor_2 = convert_element_type_default_8 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_3, input_61);  add_tensor_3 = input_61 = None
        mm_default_6 = torch.ops.aten.mm.default(view_default_8, input_83);  view_default_8 = input_83 = None
        return (convert_element_type_default_1, convert_element_type_default_4, convert_element_type_default_6, view_default_5, mm_default_5, view_default_7, mm_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
