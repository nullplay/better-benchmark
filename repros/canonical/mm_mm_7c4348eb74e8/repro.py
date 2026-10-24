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
    def forward(self, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29):
        view_default = torch.ops.aten.view.default(input_18, input_19);  input_18 = input_19 = None
        permute_default = torch.ops.aten.permute.default(input_20, [0, 1, 3, 2]);  input_20 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_21, 0.1767766952966369);  input_21 = None
        cat_default = torch.ops.aten.cat.default([mul_tensor, permute_default, view_default], 0);  mul_tensor = permute_default = view_default = None
        view_default_1 = torch.ops.aten.view.default(cat_default, input_22);  cat_default = input_22 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_1, [1, 3, 0, 2, 4]);  view_default_1 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_23);  clone_default = input_23 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_24);  view_default_2 = input_24 = None
        mm_default = torch.ops.aten.mm.default(view_default_3, input_25);  input_25 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_26);  permute_default_2 = input_26 = None
        view_default_4 = torch.ops.aten.view.default(mm_default, input_27);  mm_default = input_27 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default, input_28);  convert_element_type_default = input_28 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_29);  view_default_5 = input_29 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_6, [0, 1, 3, 2, 4, 5]);  view_default_6 = None
        return (convert_element_type_default_1, permute_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
