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
    def forward(self, input_14, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22):
        add_tensor = torch.ops.aten.add.Tensor(input_14, input_15, alpha = 1);  input_14 = input_15 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        relu_default = torch.ops.aten.relu.default(convert_element_type_default);  convert_element_type_default = None
        mean_dim = torch.ops.aten.mean.dim(relu_default, [-1, -2], True, dtype = None);  relu_default = None
        as_strided_default = torch.ops.aten.as_strided.default(mean_dim, input_16, input_17, None);  mean_dim = input_16 = input_17 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_18, torch.bfloat16);  input_18 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_19, torch.bfloat16);  input_19 = None
        convolution_default = torch.ops.aten.convolution.default(as_strided_default, convert_element_type_default_2, convert_element_type_default_1, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  as_strided_default = convert_element_type_default_2 = convert_element_type_default_1 = None
        relu_default_1 = torch.ops.aten.relu.default(convolution_default);  convolution_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_20, torch.bfloat16);  input_20 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_21, torch.bfloat16);  input_21 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        view_default = torch.ops.aten.view.default(relu_default_1, input_22);  relu_default_1 = input_22 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default_3 = view_default = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
