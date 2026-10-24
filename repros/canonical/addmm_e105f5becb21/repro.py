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
    def forward(self, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29):
        add_tensor = torch.ops.aten.add.Tensor(input_19, input_20, alpha = 1);  input_19 = input_20 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_21, input_22, alpha = 1);  input_21 = input_22 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1, alpha = 1);  convert_element_type_default = convert_element_type_default_1 = None
        relu_default = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        view_default = torch.ops.aten.view.default(input_23, input_24);  input_23 = input_24 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_25, 3);  input_25 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default, unsqueeze_default);  view_default = unsqueeze_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_26, 2);  input_26 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor, unsqueeze_default_2, alpha = 1);  mul_tensor = unsqueeze_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(convert_element_type_default_2, relu_default, alpha = 1);  convert_element_type_default_2 = relu_default = None
        relu_default_1 = torch.ops.aten.relu.default(add_tensor_4);  add_tensor_4 = None
        mean_dim = torch.ops.aten.mean.dim(relu_default_1, [-1, -2], True, dtype = None);  relu_default_1 = None
        view_default_1 = torch.ops.aten.view.default(mean_dim, input_27);  mean_dim = input_27 = None
        permute_default = torch.ops.aten.permute.default(input_28, [1, 0]);  input_28 = None
        addmm_default = torch.ops.aten.addmm.default(input_29, view_default_1, permute_default, beta = 1, alpha = 1);  input_29 = view_default_1 = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
