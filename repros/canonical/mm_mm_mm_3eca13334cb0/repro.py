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
    def forward(self, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87):
        full_default = torch.ops.aten.full.default(input_55, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_55 = None
        add_tensor = torch.ops.aten.add.Tensor(input_56, input_57, alpha = 1);  input_56 = input_57 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_58, input_59);  input_58 = input_59 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        div_scalar = torch.ops.aten.div.Scalar(input_60, 512);  input_60 = None
        mul_scalar = torch.ops.aten.mul.Scalar(input_61, 2.0);  input_61 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar);  div_scalar = mul_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_62, input_63, alpha = 1);  input_62 = input_63 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_3, input_64);  add_tensor_3 = input_64 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_65);  mul_tensor_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_65, 3);  input_65 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_66, pow_tensor_scalar);  input_66 = pow_tensor_scalar = None
        expand_default = torch.ops.aten.expand.default(mul_tensor_4, input_67, implicit = False);  mul_tensor_4 = None
        div_scalar_1 = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(input_68, 1.0);  input_68 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_1);  div_scalar_1 = mul_scalar_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_5, alpha = 1);  add_tensor_4 = mul_tensor_5 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16)
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_69, torch.bfloat16);  input_69 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_6);  convert_element_type_default = mul_tensor_6 = None
        view_default = torch.ops.aten.view.default(mul_tensor_7, input_70);  mul_tensor_7 = input_70 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_71);  permute_default = input_71 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_72);  view_default = input_72 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_73);  mm_default_1 = input_73 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_74, torch.bfloat16);  input_74 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_8);  convert_element_type_default_3 = mul_tensor_8 = None
        where_self = torch.ops.aten.where.self(input_75, full_default, mul_tensor_9);  input_75 = full_default = mul_tensor_9 = None
        view_default_2 = torch.ops.aten.view.default(where_self, input_76);  where_self = input_76 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_77);  permute_default_1 = input_77 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_2, input_78);  view_default_2 = input_78 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_3, input_67);  mm_default_3 = input_67 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, input_79);  input_79 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, input_80);  input_80 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_10, input_81);  mul_tensor_10 = input_81 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_12, alpha = 1);  add_tensor_5 = mul_tensor_12 = None
        div_scalar_2 = torch.ops.aten.div.Scalar(input_82, 512);  input_82 = None
        mul_scalar_2 = torch.ops.aten.mul.Scalar(input_83, 2.0);  input_83 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(div_scalar_2, mul_scalar_2);  div_scalar_2 = mul_scalar_2 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, mul_tensor_13, alpha = 1);  add_tensor_6 = mul_tensor_13 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16)
        mul_tensor_14 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, input_84);  convert_element_type_default_7 = input_84 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(input_85, input_86);  input_85 = input_86 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_7, mul_tensor_15, alpha = 1);  add_tensor_7 = mul_tensor_15 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_8, input_87, alpha = 1);  add_tensor_8 = input_87 = None
        return (convert_element_type_default_2, convert_element_type_default_5, convert_element_type_default_6, mul_tensor_11, mul_tensor_14, add_tensor_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
