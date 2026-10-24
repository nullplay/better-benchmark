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
    def forward(self, input_22, input_23, input_24, input_25, input_26, input_27, input_28):
        add_tensor = torch.ops.aten.add.Tensor(input_22, input_23, alpha = 1);  input_22 = input_23 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_1, 3, alpha = 1)
        clamp_min_default = torch.ops.aten.clamp_min.default(add_tensor_1, 0);  add_tensor_1 = None
        clamp_max_default = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, clamp_max_default);  convert_element_type_default_1 = clamp_max_default = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor, 6);  mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mean_dim = torch.ops.aten.mean.dim(convert_element_type_default_2, [-1, -2], True, dtype = None);  convert_element_type_default_2 = None
        view_default = torch.ops.aten.view.default(mean_dim, input_24);  mean_dim = input_24 = None
        permute_default = torch.ops.aten.permute.default(input_25, [1, 0]);  input_25 = None
        addmm_default = torch.ops.aten.addmm.default(input_26, view_default, permute_default, beta = 1, alpha = 1);  input_26 = view_default = permute_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(addmm_default, torch.float32);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default_3, 3, alpha = 1)
        clamp_min_default_1 = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default_1 = torch.ops.aten.clamp_max.default(clamp_min_default_1, 6);  clamp_min_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, clamp_max_default_1);  convert_element_type_default_3 = clamp_max_default_1 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(mul_tensor_1, 6);  mul_tensor_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None
        permute_default_1 = torch.ops.aten.permute.default(input_27, [1, 0]);  input_27 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_28, convert_element_type_default_4, permute_default_1, beta = 1, alpha = 1);  input_28 = convert_element_type_default_4 = permute_default_1 = None
        return (addmm_default_1,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
