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
    def forward(self, input_10, input_11, input_12, input_13, input_14, input_15, input_16, input_17, input_18):
        add_tensor = torch.ops.aten.add.Tensor(input_10, input_11, alpha = 1);  input_10 = input_11 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        relu_default = torch.ops.aten.relu.default(convert_element_type_default);  convert_element_type_default = None
        mean_dim = torch.ops.aten.mean.dim(relu_default, [-1, -2], True, dtype = None);  relu_default = None
        as_strided_default = torch.ops.aten.as_strided.default(mean_dim, input_12, input_13, None);  mean_dim = input_12 = input_13 = None
        convolution_default = torch.ops.aten.convolution.default(as_strided_default, input_14, input_15, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  as_strided_default = input_14 = input_15 = None
        relu_default_1 = torch.ops.aten.relu.default(convolution_default);  convolution_default = None
        view_default = torch.ops.aten.view.default(relu_default_1, input_16);  relu_default_1 = input_16 = None
        permute_default = torch.ops.aten.permute.default(input_17, [1, 0]);  input_17 = None
        addmm_default = torch.ops.aten.addmm.default(input_18, view_default, permute_default, beta = 1, alpha = 1);  input_18 = view_default = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
