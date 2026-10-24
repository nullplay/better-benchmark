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
    def forward(self, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_26, torch.bfloat16);  input_26 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_27, torch.bfloat16);  input_27 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_28, torch.bfloat16);  input_28 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_29);  convert_element_type_default_2 = input_29 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_30);  addmm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_31, torch.bfloat16);  input_31 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_32, torch.bfloat16);  input_32 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = view_default = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default_1, input_30);  addmm_default_1 = input_30 = None
        view_default_3 = torch.ops.aten.view.default(view_default_1, input_33);  view_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(view_default_2, input_33);  view_default_2 = input_33 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 2]);  permute_default_2 = None
        mul_scalar = torch.ops.aten.mul.Scalar(permute_default_4, 0.3535533905932738);  permute_default_4 = None
        expand_default = torch.ops.aten.expand.default(mul_scalar, input_34, implicit = False);  mul_scalar = input_34 = None
        view_default_5 = torch.ops.aten.view.default(input_35, input_36);  input_35 = input_36 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_3, input_37, implicit = False);  permute_default_3 = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_38);  clone_default = input_38 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_5, view_default_6);  view_default_5 = None
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_37);  bmm_default = input_37 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1]);  view_default_6 = None
        return (expand_default, view_default_7, permute_default_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
