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
    def forward(self, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_36, torch.bfloat16);  input_36 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_37);  convert_element_type_default = input_37 = None
        addmm_default = torch.ops.aten.addmm.default(input_38, view_default, input_39, beta = 1, alpha = 1);  input_38 = input_39 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_40, torch.bfloat16);  input_40 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default_1 = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default_1, input_42)
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_1, input_43);  view_default_1 = input_43 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, input_44, alpha = 1);  mul_tensor = input_44 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_45, torch.bfloat16);  input_45 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_46, torch.bfloat16);  input_46 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_5, input_47);  convert_element_type_default_5 = input_47 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_2, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_2, input_42);  addmm_default_2 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_48);  view_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_49, torch.bfloat16);  input_49 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_2, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_3, input_42);  addmm_default_3 = input_42 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_48);  view_default_5 = input_48 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_51, view_default, input_52, beta = 1, alpha = 1);  input_51 = view_default = input_52 = None
        mul_scalar = torch.ops.aten.mul.Scalar(permute_default_2, 0.4204482076268573);  permute_default_2 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 1, 3, 2]);  permute_default_4 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(permute_default_5, 0.4204482076268573);  permute_default_5 = None
        expand_default = torch.ops.aten.expand.default(mul_scalar, input_53, implicit = False);  mul_scalar = input_53 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_54);  clone_default = input_54 = None
        expand_default_1 = torch.ops.aten.expand.default(mul_scalar_1, input_55, implicit = False);  mul_scalar_1 = input_55 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_7, input_56);  input_56 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1]);  view_default_7 = None
        return (addmm_default, addmm_default_1, view_default_2, addmm_default_4, expand_default_1, bmm_default, permute_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
