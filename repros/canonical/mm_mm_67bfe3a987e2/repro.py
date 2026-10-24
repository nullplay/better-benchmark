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
    def forward(self, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26):
        view_default = torch.ops.aten.view.default(input_17, input_18);  input_17 = None
        div_tensor = torch.ops.aten.div.Tensor(input_19, 0.8999999985098839);  input_19 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default, div_tensor);  view_default = div_tensor = None
        view_default_1 = torch.ops.aten.view.default(mul_tensor, input_20);  mul_tensor = input_20 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 2, 4, 5]);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(permute_default, input_18);  permute_default = input_18 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_21);  view_default_2 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_22);  view_default_3 = input_22 = None
        mm_default = torch.ops.aten.mm.default(view_default_4, input_23);  input_23 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_1, input_24);  permute_default_1 = input_24 = None
        view_default_5 = torch.ops.aten.view.default(mm_default, input_21);  mm_default = input_21 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_25);  view_default_5 = input_25 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_26);  clone_default = input_26 = None
        return (convert_element_type_default, view_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
