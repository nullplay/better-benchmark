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
    def forward(self, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89):
        add_tensor = torch.ops.aten.add.Tensor(input_61, input_62, alpha = 1);  input_61 = input_62 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_63, torch.float32);  input_63 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default, alpha = 1);  add_tensor = convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(input_64, input_65);  input_64 = input_65 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_1, alpha = 1);  add_tensor_1 = convert_element_type_default_1 = None
        permute_default = torch.ops.aten.permute.default(input_66, [1, 0]);  input_66 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_67);  permute_default = input_67 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_2, alpha = 1);  add_tensor_2 = convert_element_type_default_2 = None
        view_default = torch.ops.aten.view.default(input_68, input_69);  input_68 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_70);  permute_default_1 = input_70 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, convert_element_type_default_3, alpha = 1);  add_tensor_3 = convert_element_type_default_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(input_71, input_72, alpha = 1);  input_71 = input_72 = None
        view_default_1 = torch.ops.aten.view.default(add_tensor_5, input_69);  add_tensor_5 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_1, input_73)
        permute_default_2 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_4 = torch.ops.aten.mm.default(permute_default_2, input_74);  permute_default_2 = input_74 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_4, alpha = 1);  add_tensor_4 = convert_element_type_default_4 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_75, input_76, alpha = 1);  input_75 = input_76 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_77, sub_tensor);  input_77 = sub_tensor = None
        view_default_2 = torch.ops.aten.view.default(input_78, input_79);  input_78 = input_79 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, input_80);  input_80 = None
        tanh_default = torch.ops.aten.tanh.default(input_81);  input_81 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, add_tensor_7);  convert_element_type_default_5 = add_tensor_7 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(1, input_82, alpha = 1);  input_82 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, sub_tensor_1);  mul_tensor_1 = sub_tensor_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 0.7978845608028654);  mul_tensor_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 0.044715);  mul_tensor_4 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_83, 2.0);  input_83 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_scalar);  mul_tensor_5 = mul_scalar = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(convert_element_type_default_7, convert_element_type_default_8, alpha = 1);  convert_element_type_default_7 = convert_element_type_default_8 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 0.5);  convert_element_type_default_6 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_8, mul_tensor_7, alpha = 1);  add_tensor_8 = mul_tensor_7 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_9, input_69);  add_tensor_9 = input_69 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_3, input_73);  input_73 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mm_default_6 = torch.ops.aten.mm.default(permute_default_3, input_84);  permute_default_3 = input_84 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_5, input_85);  mm_default_5 = input_85 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default_9, alpha = 1);  mul_tensor = convert_element_type_default_9 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_6, convert_element_type_default_10, alpha = 1);  add_tensor_6 = convert_element_type_default_10 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_tensor_10, input_86);  input_86 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 4096)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True, dtype = None)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_87, input_88);  input_87 = input_88 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_8, mul_tensor_10);  mul_tensor_8 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(add_tensor_10, mul_tensor_10);  mul_tensor_10 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1], False, dtype = None);  mul_tensor_12 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(add_tensor_10, [0, 1], False, dtype = None);  add_tensor_10 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(input_89, sum_dim_int_list_2, alpha = 1);  input_89 = sum_dim_int_list_2 = None
        return (mm_default_3, add_tensor_11, mul_tensor_9, sum_dim_int_list, mul_tensor_11, sum_dim_int_list_1, add_tensor_12)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
