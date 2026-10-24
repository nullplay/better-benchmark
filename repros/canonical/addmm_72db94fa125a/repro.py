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
    def forward(self, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20):
        mean_dim = torch.ops.aten.mean.dim(input_13, [-1, -2], True, dtype = None);  input_13 = None
        as_strided_default = torch.ops.aten.as_strided.default(mean_dim, input_14, input_15, None);  mean_dim = input_14 = input_15 = None
        convolution_default = torch.ops.aten.convolution.default(as_strided_default, input_16, input_17, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  as_strided_default = input_16 = input_17 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(convolution_default, torch.float32);  convolution_default = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, 3, alpha = 1)
        clamp_min_default = torch.ops.aten.clamp_min.default(add_tensor, 0);  add_tensor = None
        clamp_max_default = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, clamp_max_default);  convert_element_type_default = clamp_max_default = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor, 6);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_18);  convert_element_type_default_1 = input_18 = None
        permute_default = torch.ops.aten.permute.default(input_19, [1, 0]);  input_19 = None
        addmm_default = torch.ops.aten.addmm.default(input_20, view_default, permute_default, beta = 1, alpha = 1);  input_20 = view_default = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
