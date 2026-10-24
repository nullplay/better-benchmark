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
    def forward(self, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48):
        add_tensor = torch.ops.aten.add.Tensor(input_28, input_29, alpha = 1);  input_28 = input_29 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_30, input_31);  input_30 = input_31 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_32, input_33, alpha = 1);  input_32 = input_33 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_34, sub_tensor);  input_34 = sub_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_35, input_36);  input_35 = input_36 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_37, input_38, alpha = 1);  input_37 = input_38 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_39, sub_tensor_2);  input_39 = sub_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_40, torch.bfloat16);  input_40 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_4);  convert_element_type_default = mul_tensor_4 = None
        view_default = torch.ops.aten.view.default(mul_tensor_5, input_41);  mul_tensor_5 = input_41 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_42);  input_42 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_43);  permute_default = input_43 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_44);  mm_default = input_44 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_45, 0.5);  input_45 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_46, input_47);  input_46 = input_47 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_6, mul_tensor_7, alpha = 1);  mul_tensor_6 = mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, add_tensor_4);  convert_element_type_default_3 = add_tensor_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_4, input_48);  convert_element_type_default_4 = input_48 = None
        return (convert_element_type_default_2, view_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
