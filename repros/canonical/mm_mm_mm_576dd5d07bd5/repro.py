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
    def forward(self, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88):
        mul_tensor = torch.ops.aten.mul.Tensor(input_66, input_67);  input_66 = input_67 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_68, input_69);  input_68 = input_69 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1, alpha = 1);  mul_tensor = mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(input_70, input_71);  input_70 = input_71 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_72);  view_default = input_72 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_73);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_74, torch.float32);  input_74 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_2);  convert_element_type_default = mul_tensor_2 = None
        view_default_2 = torch.ops.aten.view.default(input_75, input_73);  input_75 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 3.0);  convert_element_type_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_2, mul_tensor_5, alpha = 1);  view_default_2 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_1, 0.7978845608028654);  add_tensor_1 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_6);  mul_tensor_6 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_2)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_7);  mul_tensor_7 = None
        view_default_3 = torch.ops.aten.view.default(input_76, input_73);  input_76 = input_73 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_3, view_default_3);  mul_tensor_3 = view_default_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_3, input_77);  convert_element_type_default_3 = None
        permute_default = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_78);  permute_default = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, input_79);  view_default_4 = input_79 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_80);  mm_default_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_9, add_tensor_2);  mul_tensor_9 = add_tensor_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_11, torch.bfloat16);  mul_tensor_11 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor = torch.ops.aten.sub.Tensor(1, mul_tensor_12, alpha = 1);  mul_tensor_12 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_10, sub_tensor);  mul_tensor_10 = sub_tensor = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, 0.7978845608028654);  mul_tensor_13 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 0.5);  convert_element_type_default_6 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_81, mul_tensor_15, alpha = 1);  input_81 = mul_tensor_15 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_3, input_77);  add_tensor_3 = input_77 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_6, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_78);  permute_default_1 = input_78 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_6, input_82);  view_default_6 = input_82 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_4, input_80);  mm_default_4 = input_80 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(convert_element_type_default_4, convert_element_type_default_7, alpha = 1);  convert_element_type_default_4 = convert_element_type_default_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(add_tensor_4, input_83);  input_83 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(input_84, input_85)
        mul_tensor_18 = torch.ops.aten.mul.Tensor(add_tensor_4, mul_tensor_17);  add_tensor_4 = mul_tensor_17 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1], True, dtype = torch.float32);  mul_tensor_18 = None
        view_default_8 = torch.ops.aten.view.default(sum_dim_int_list, input_86);  sum_dim_int_list = input_86 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(mul_tensor_16, input_84)
        mul_tensor_20 = torch.ops.aten.mul.Tensor(mul_tensor_16, input_85);  mul_tensor_16 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [2], True, dtype = torch.float32);  mul_tensor_19 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_20, alpha = 1);  add_tensor = mul_tensor_20 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(input_85, 3);  input_85 = None
        mul_scalar = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar_1);  mul_scalar = pow_tensor_scalar_1 = None
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(input_84, 1.0);  input_84 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(pow_tensor_scalar_2, 2.0);  pow_tensor_scalar_2 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(input_87, mul_scalar_1);  input_87 = mul_scalar_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_22, alpha = 1);  add_tensor_5 = mul_tensor_22 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16)
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, input_88, alpha = 1);  add_tensor_6 = input_88 = None
        return (convert_element_type_default_5, mul_tensor_14, convert_element_type_default_8, view_default_8, mul_tensor_21, convert_element_type_default_9, add_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
