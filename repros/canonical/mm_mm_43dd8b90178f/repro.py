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
    def forward(self, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        view_default = torch.ops.aten.view.default(input_24, input_25);  input_24 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_26, input_27);  input_26 = input_27 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, mul_tensor, alpha = 1);  view_default = mul_tensor = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_28);  add_tensor = input_28 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_29, input_30);  input_29 = input_30 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_31, input_32, alpha = 1);  input_31 = input_32 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_1, alpha = 1);  sub_tensor = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_33, sub_tensor_1);  input_33 = sub_tensor_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_2, alpha = 1);  view_default_1 = mul_tensor_2 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_25);  add_tensor_1 = input_25 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_2, torch.bfloat16);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default, input_34);  convert_element_type_default = input_34 = None
        mm_default = torch.ops.aten.mm.default(view_default_3, input_35);  input_35 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_36);  permute_default = input_36 = None
        view_default_4 = torch.ops.aten.view.default(mm_default, input_37);  mm_default = input_37 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_38, 0.5);  input_38 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_39, input_40);  input_39 = input_40 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_3, mul_tensor_4, alpha = 1);  mul_tensor_3 = mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, add_tensor_2);  convert_element_type_default_2 = add_tensor_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_3, input_41);  convert_element_type_default_3 = input_41 = None
        return (convert_element_type_default_1, view_default_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
