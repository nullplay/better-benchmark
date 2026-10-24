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
    def forward(self, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22):
        mm_default = torch.ops.aten.mm.default(input_13, input_14);  input_14 = None
        permute_default = torch.ops.aten.permute.default(input_13, [1, 0]);  input_13 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_15);  permute_default = input_15 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_16);  convert_element_type_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default, 3);  view_default = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default = torch.ops.aten.full.default(input_17, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_17 = None
        as_strided_scatter_default = torch.ops.aten.as_strided_scatter.default(full_default, squeeze_dim_1, input_18, input_19, input_20);  full_default = squeeze_dim_1 = input_18 = input_19 = None
        as_strided_default = torch.ops.aten.as_strided.default(as_strided_scatter_default, input_16, input_21, input_20);  as_strided_scatter_default = input_16 = input_21 = input_20 = None
        expand_default = torch.ops.aten.expand.default(as_strided_default, input_22, implicit = False);  as_strided_default = input_22 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        return (convert_element_type_default_1, div_scalar)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
