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
    def forward(self, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56):
        rsqrt_default = torch.ops.aten.rsqrt.default(input_41);  input_41 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_42, input_43, alpha = 1);  input_42 = input_43 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_44);  mul_tensor = input_44 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, input_45, alpha = 1);  mul_tensor_1 = input_45 = None
        permute_default = torch.ops.aten.permute.default(add_tensor, [1, 0, 2]);  add_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_46, torch.bfloat16);  input_46 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_47, torch.bfloat16);  input_47 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(permute_default, torch.bfloat16);  permute_default = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        clone_default = torch.ops.aten.clone.default(convert_element_type_default_2, memory_format = torch.contiguous_format);  convert_element_type_default_2 = None
        view_default = torch.ops.aten.view.default(clone_default, input_48);  clone_default = input_48 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default_1);  permute_default_1 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_49);  mm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, convert_element_type_default, alpha = 1);  view_default_1 = convert_element_type_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, permute_default_2);  permute_default_2 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_49);  mm_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_2, convert_element_type_default_3, alpha = 1);  view_default_2 = convert_element_type_default_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_52, torch.bfloat16);  input_52 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_53, torch.bfloat16);  input_53 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_6, [1, 0]);  convert_element_type_default_6 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default, permute_default_3);  view_default = permute_default_3 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_2, input_49);  mm_default_2 = input_49 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_3, convert_element_type_default_5, alpha = 1);  view_default_3 = convert_element_type_default_5 = None
        div_tensor = torch.ops.aten.div.Tensor(add_tensor_1, 8.0);  add_tensor_1 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_2, input_54);  add_tensor_2 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_4, [1, 0, 2, 3]);  view_default_4 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 2, 1, 3]);  permute_default_4 = None
        view_default_5 = torch.ops.aten.view.default(permute_default_5, input_55);  permute_default_5 = None
        view_default_6 = torch.ops.aten.view.default(div_tensor, input_54);  div_tensor = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_6, [1, 0, 2, 3]);  view_default_6 = None
        permute_default_7 = torch.ops.aten.permute.default(permute_default_6, [0, 2, 1, 3]);  permute_default_6 = None
        view_default_7 = torch.ops.aten.view.default(permute_default_7, input_55);  permute_default_7 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_56);  view_default_7 = input_56 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_3, input_54);  add_tensor_3 = input_54 = None
        permute_default_8 = torch.ops.aten.permute.default(view_default_9, [1, 0, 2, 3]);  view_default_9 = None
        permute_default_9 = torch.ops.aten.permute.default(permute_default_8, [0, 2, 1, 3]);  permute_default_8 = None
        view_default_10 = torch.ops.aten.view.default(permute_default_9, input_55);  permute_default_9 = input_55 = None
        return (view_default_5, view_default_8, view_default_10)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
