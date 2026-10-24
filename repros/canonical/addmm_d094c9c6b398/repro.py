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
    def forward(self, input_14, input_15, input_16, input_17, input_18, input_19, input_20):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_14, torch.float32);  input_14 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5);  convert_element_type_default = None
        erf_default = torch.ops.aten.erf.default(input_15);  input_15 = None
        add_tensor = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.7015043497085571);  convert_element_type_default_1 = None
        mean_dim = torch.ops.aten.mean.dim(mul_tensor_2, [-1, -2], True, dtype = None);  mul_tensor_2 = None
        as_strided_default = torch.ops.aten.as_strided.default(mean_dim, input_16, input_17, None);  mean_dim = input_16 = input_17 = None
        view_default = torch.ops.aten.view.default(as_strided_default, input_18);  as_strided_default = input_18 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_19, torch.bfloat16);  input_19 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_20, torch.bfloat16);  input_20 = None
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
