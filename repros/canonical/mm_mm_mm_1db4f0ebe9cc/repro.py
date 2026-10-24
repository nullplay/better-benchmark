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
    def forward(self, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33):
        view_default = torch.ops.aten.view.default(input_25, input_26);  input_25 = input_26 = None
        view_default_1 = torch.ops.aten.view.default(input_27, input_28);  input_27 = input_28 = None
        mul_scalar = torch.ops.aten.mul.Scalar(view_default_1, 0.334370152488211);  view_default_1 = None
        permute_default = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        permute_default_1 = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_29);  clone_default = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(permute_default_2, input_29);  permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(view_default_2, input_30);  view_default_2 = None
        mm_default = torch.ops.aten.mm.default(view_default_4, input_31);  input_31 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_3, input_32);  permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(mm_default, input_29);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        clone_default_1 = torch.ops.aten.clone.default(view_default_3, memory_format = torch.contiguous_format);  view_default_3 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_30);  clone_default_1 = input_30 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_6, input_33);  input_33 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_6, [1, 0]);  view_default_6 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_4, input_32);  permute_default_4 = input_32 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_2, input_29);  mm_default_2 = input_29 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_2, alpha = 1);  convert_element_type_default = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        return (convert_element_type_default_1, add_tensor, convert_element_type_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
