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
    def forward(self, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82):
        add_tensor = torch.ops.aten.add.Tensor(input_58, input_59, alpha = 1);  input_58 = input_59 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_60, input_61);  input_60 = input_61 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_62, input_63, alpha = 1);  input_62 = input_63 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_64, sub_tensor);  input_64 = sub_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_65, input_66);  input_65 = input_66 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_67, input_68, alpha = 1);  input_67 = input_68 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_69, sub_tensor_2);  input_69 = sub_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_70, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_70, [2], True, dtype = None);  input_70 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_71, [2], True, dtype = None);  input_71 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_72, sum_dim_int_list_1);  input_72 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_5, alpha = 1);  sub_tensor_3 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_73, sub_tensor_4);  input_73 = sub_tensor_4 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6, alpha = 1);  add_tensor_3 = mul_tensor_6 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_74, torch.bfloat16);  input_74 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_7);  convert_element_type_default = mul_tensor_7 = None
        view_default = torch.ops.aten.view.default(mul_tensor_8, input_75);  mul_tensor_8 = input_75 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_76);  input_76 = None
        mm_default_1 = torch.ops.aten.mm.default(input_77, view_default);  input_77 = view_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_2, input_78);  convert_element_type_default_2 = None
        view_default_2 = torch.ops.aten.view.default(input_79, input_78);  input_79 = input_78 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_default_1, mul_tensor_9);  mul_tensor_9 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 3.0)
        mul_tensor_11 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_2, mul_tensor_11, alpha = 1);  view_default_2 = mul_tensor_11 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(add_tensor_5, 0.7978845608028654);  add_tensor_5 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_12);  mul_tensor_12 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_13 = torch.ops.aten.mul.Tensor(view_default_1, add_tensor_6);  view_default_1 = add_tensor_6 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_13, torch.bfloat16);  mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(1, mul_tensor_14, alpha = 1);  mul_tensor_14 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_10, sub_tensor_5);  mul_tensor_10 = sub_tensor_5 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_15, 0.7978845608028654);  mul_tensor_15 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16)
        mul_tensor_17 = torch.ops.aten.mul.Tensor(mul_tensor_16, 0.044715);  mul_tensor_16 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2.0);  convert_element_type_default_4 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_17, mul_scalar);  mul_tensor_17 = mul_scalar = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_18, torch.bfloat16);  mul_tensor_18 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(convert_element_type_default_6, convert_element_type_default_7, alpha = 1);  convert_element_type_default_6 = convert_element_type_default_7 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 0.5);  convert_element_type_default_5 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_19, alpha = 1);  add_tensor_7 = mul_tensor_19 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_8, input_80);  add_tensor_8 = input_80 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_3, input_81);  input_81 = None
        mm_default_3 = torch.ops.aten.mm.default(input_82, view_default_3);  input_82 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = torch.float32);  view_default_3 = None
        return (convert_element_type_default_3, mm_default_2, mm_default_3, sum_dim_int_list_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
