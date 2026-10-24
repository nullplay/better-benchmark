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
    def forward(self, input_14, input_15, input_16, input_17, input_18):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_14, torch.float32);  input_14 = None
        clamp_min_default = torch.ops.aten.clamp_min.default(convert_element_type_default, 0.0);  convert_element_type_default = None
        clamp_max_default = torch.ops.aten.clamp_max.default(clamp_min_default, 6.0);  clamp_min_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(clamp_max_default, torch.bfloat16);  clamp_max_default = None
        mean_dim = torch.ops.aten.mean.dim(convert_element_type_default_1, [-1, -2], True, dtype = None);  convert_element_type_default_1 = None
        view_default = torch.ops.aten.view.default(mean_dim, input_15);  mean_dim = input_15 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_16, torch.bfloat16);  input_16 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.2);  convert_element_type_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.25);  mul_tensor = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_17, torch.bfloat16);  input_17 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_18, torch.bfloat16);  input_18 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_3, mul_tensor_1, permute_default, beta = 1, alpha = 1);  convert_element_type_default_3 = mul_tensor_1 = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
