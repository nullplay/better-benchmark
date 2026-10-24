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
    def forward(self, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28):
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_21, 1);  input_21 = None
        expand_default = torch.ops.aten.expand.default(unsqueeze_default, input_22, implicit = False);  unsqueeze_default = input_22 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = None);  expand_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(clone_default, torch.bfloat16);  clone_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(convert_element_type_default, 3);  convert_element_type_default = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 4);  unsqueeze_default_1 = None
        view_default = torch.ops.aten.view.default(unsqueeze_default_2, input_23);  unsqueeze_default_2 = input_23 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default, 0);  view_default = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(input_24, 3);  input_24 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 4);  unsqueeze_default_3 = None
        view_default_1 = torch.ops.aten.view.default(unsqueeze_default_4, input_25);  unsqueeze_default_4 = input_25 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        mm_default = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_1);  squeeze_dim = squeeze_dim_1 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(unsqueeze_default_5, input_26);  unsqueeze_default_5 = input_26 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 4, 2]);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(permute_default, input_27);  permute_default = input_27 = None
        unsqueeze_default_6 = torch.ops.aten.unsqueeze.default(view_default_3, 4);  view_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(unsqueeze_default_6, [1, 2, 4, 0, 3]);  unsqueeze_default_6 = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default_1, [0, 1, 4, 3, 2]);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_2, input_28);  permute_default_2 = input_28 = None
        return (view_default_4,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
