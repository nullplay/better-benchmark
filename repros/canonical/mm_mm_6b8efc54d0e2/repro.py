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
        mm_default = torch.ops.aten.mm.default(input_11, input_12);  input_12 = None
        permute_default = torch.ops.aten.permute.default(input_11, [1, 0]);  input_11 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_13);  permute_default = input_13 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_14, torch.bfloat16);  input_14 = None
        div_scalar = torch.ops.aten.div.Scalar(convert_element_type_default_1, 0.8);  convert_element_type_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(mm_default, div_scalar);  mm_default = div_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(mul_tensor, 2);  mul_tensor = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 3);  unsqueeze_default = None
        expand_default = torch.ops.aten.expand.default(unsqueeze_default_1, input_15, implicit = False);  unsqueeze_default_1 = input_15 = None
        div_scalar_1 = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        return (convert_element_type_default, div_scalar_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
