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
    def forward(self, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24):
        convolution_default = torch.ops.aten.convolution.default(input_17, input_18, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  input_17 = input_18 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_19, -1);  input_19 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convolution_default, unsqueeze_default, alpha = 1);  convolution_default = unsqueeze_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_20, -1);  input_20 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, -1);  unsqueeze_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_2);  sub_tensor = unsqueeze_default_2 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(input_21, -1);  input_21 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, -1);  unsqueeze_default_3 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_4);  mul_tensor = unsqueeze_default_4 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(input_22, -1);  input_22 = None
        unsqueeze_default_6 = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, -1);  unsqueeze_default_5 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_6, alpha = 1);  mul_tensor_1 = unsqueeze_default_6 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        relu_default = torch.ops.aten.relu.default(convert_element_type_default);  convert_element_type_default = None
        mean_dim = torch.ops.aten.mean.dim(relu_default, [2, 3], False, dtype = None);  relu_default = None
        permute_default = torch.ops.aten.permute.default(input_23, [1, 0]);  input_23 = None
        addmm_default = torch.ops.aten.addmm.default(input_24, mean_dim, permute_default, beta = 1, alpha = 1);  input_24 = mean_dim = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
