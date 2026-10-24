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
    def forward(self, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27):
        mul_tensor = torch.ops.aten.mul.Tensor(input_17, input_18);  input_17 = input_18 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_19, -1);  input_19 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, unsqueeze_default, alpha = 1);  mul_tensor = unsqueeze_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_20, input_21);  input_20 = input_21 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_22, -1);  input_22 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_1, alpha = 1);  mul_tensor_1 = unsqueeze_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1, alpha = 1);  convert_element_type_default = convert_element_type_default_1 = None
        relu_default = torch.ops.aten.relu.default(add_tensor_2);  add_tensor_2 = None
        mean_dim = torch.ops.aten.mean.dim(relu_default, [-1, -2], True, dtype = None);  relu_default = None
        as_strided_default = torch.ops.aten.as_strided.default(mean_dim, input_23, input_24, None);  mean_dim = input_23 = input_24 = None
        view_default = torch.ops.aten.view.default(as_strided_default, input_25);  as_strided_default = input_25 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_26, torch.bfloat16);  input_26 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_27, torch.bfloat16);  input_27 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_2, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default_2 = view_default = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
