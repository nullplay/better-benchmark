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
    def forward(self, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18):
        mm_default = torch.ops.aten.mm.default(input_11, input_12);  input_12 = None
        permute_default = torch.ops.aten.permute.default(input_11, [1, 0]);  input_11 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_13);  permute_default = input_13 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(mm_default, input_14);  mm_default = input_14 = None
        expand_default = torch.ops.aten.expand.default(view_default, input_15, implicit = False);  view_default = input_15 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        full_default = torch.ops.aten.full.default(input_16, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_16 = None
        where_self = torch.ops.aten.where.self(input_17, full_default, div_scalar);  input_17 = full_default = div_scalar = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(where_self, torch.float32)
        add_tensor = torch.ops.aten.add.Tensor(where_self, input_18, alpha = 1);  where_self = input_18 = None
        return (convert_element_type_default, convert_element_type_default_1, add_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
