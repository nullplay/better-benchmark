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
    def forward(self, input_8, input_9, input_10, input_11, input_12, input_13, input_14):
        expand_default = torch.ops.aten.expand.default(input_8, input_9, implicit = False);  input_8 = input_9 = None
        view_default = torch.ops.aten.view.default(expand_default, input_10);  expand_default = input_10 = None
        permute_default = torch.ops.aten.permute.default(input_11, [1, 0]);  input_11 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_12);  mm_default = input_12 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_1, input_13, alpha = 1);  view_default_1 = input_13 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor, input_14);  add_tensor = input_14 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        return (permute_default_1,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
