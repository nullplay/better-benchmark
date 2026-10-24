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
    def forward(self, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35):
        view_default = torch.ops.aten.view.default(input_23, input_24);  input_23 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_25, input_26);  input_25 = input_26 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_27, input_28, alpha = 1);  input_27 = input_28 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor, alpha = 1);  sub_tensor = mul_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(input_29, 768);  input_29 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_1);  div_tensor = sub_tensor_1 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, mul_tensor_1, alpha = 1);  view_default = mul_tensor_1 = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_30);  add_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_31, torch.bfloat16);  input_31 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_2);  convert_element_type_default = mul_tensor_2 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_3, input_24);  mul_tensor_3 = None
        mm_default = torch.ops.aten.mm.default(view_default_2, input_32);  input_32 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        permute_default_1 = torch.ops.aten.permute.default(input_33, [0, 2, 1, 3]);  input_33 = None
        view_default_3 = torch.ops.aten.view.default(permute_default_1, input_34);  permute_default_1 = input_34 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_24);  view_default_3 = input_24 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, view_default_4);  permute_default = view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default, input_30);  mm_default = input_30 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_35);  view_default_5 = input_35 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        return (convert_element_type_default_2, permute_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
