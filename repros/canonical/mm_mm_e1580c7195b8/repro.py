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
    def forward(self, input_11, input_12, input_13, input_14, input_15):
        getitem = input_11[0];  input_11 = None
        permute_default = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default = torch.ops.aten.view.default(permute_default, input_12);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_13);  view_default = input_13 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_14);  input_14 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_1, input_15);  permute_default_1 = input_15 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_12);  mm_default = input_12 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [0], True, dtype = torch.float32);  convert_element_type_default = None
        return (convert_element_type_default_1, sum_dim_int_list)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
