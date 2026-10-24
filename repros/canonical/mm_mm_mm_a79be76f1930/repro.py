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
    def forward(self, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85):
        add_tensor = torch.ops.aten.add.Tensor(input_60, input_61, alpha = 1);  input_60 = input_61 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_62, input_63);  input_62 = input_63 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        div_scalar = torch.ops.aten.div.Scalar(input_64, 512);  input_64 = None
        mul_scalar = torch.ops.aten.mul.Scalar(input_65, 2.0);  input_65 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar);  div_scalar = mul_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_66, input_67, alpha = 1);  input_66 = input_67 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_3, input_68);  add_tensor_3 = input_68 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_69);  mul_tensor_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_69, 3);  input_69 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_70, pow_tensor_scalar);  input_70 = pow_tensor_scalar = None
        expand_default = torch.ops.aten.expand.default(mul_tensor_4, input_71, implicit = False);  mul_tensor_4 = None
        div_scalar_1 = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(input_72, 1.0);  input_72 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_1);  div_scalar_1 = mul_scalar_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_5, alpha = 1);  add_tensor_4 = mul_tensor_5 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_73, torch.bfloat16);  input_73 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_6);  convert_element_type_default = mul_tensor_6 = None
        view_default = torch.ops.aten.view.default(mul_tensor_7, input_74);  mul_tensor_7 = input_74 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_75);  permute_default = input_75 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_76);  view_default = input_76 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_77);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_78, torch.float32);  input_78 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_8);  convert_element_type_default_2 = mul_tensor_8 = None
        view_default_2 = torch.ops.aten.view.default(input_79, input_77);  input_79 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_2, input_80, alpha = 1);  view_default_2 = input_80 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(add_tensor_6, 0.7978845608028654);  add_tensor_6 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_11);  mul_tensor_11 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_10, add_tensor_7)
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_12);  mul_tensor_12 = None
        view_default_3 = torch.ops.aten.view.default(input_81, input_77);  input_81 = input_77 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_9, view_default_3);  mul_tensor_9 = view_default_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_13, torch.bfloat16);  mul_tensor_13 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_5, input_82);  convert_element_type_default_5 = input_82 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_83);  permute_default_1 = input_83 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_4, input_84);  view_default_4 = input_84 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_3, input_71);  mm_default_3 = input_71 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_tensor_10);  mul_tensor_10 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_14, add_tensor_7);  mul_tensor_14 = add_tensor_7 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor = torch.ops.aten.sub.Tensor(1, mul_tensor_17, alpha = 1);  mul_tensor_17 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_15, sub_tensor);  mul_tensor_15 = sub_tensor = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(mul_tensor_18, 0.7978845608028654);  mul_tensor_18 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_19, torch.bfloat16)
        mul_tensor_20 = torch.ops.aten.mul.Tensor(mul_tensor_19, 0.044715);  mul_tensor_19 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 0.5);  convert_element_type_default_6 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(input_85, mul_tensor_21, alpha = 1);  input_85 = mul_tensor_21 = None
        return (convert_element_type_default_3, mm_default_2, view_default_5, convert_element_type_default_7, mul_tensor_20, add_tensor_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
