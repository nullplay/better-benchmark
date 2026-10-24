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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37):
        add_tensor = torch.ops.aten.add.Tensor(input_30, input_31, alpha = 1);  input_30 = input_31 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_1, 3, alpha = 1)
        clamp_min_default = torch.ops.aten.clamp_min.default(add_tensor_1, 0);  add_tensor_1 = None
        clamp_max_default = torch.ops.aten.clamp_max.default(clamp_min_default, 6);  clamp_min_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, clamp_max_default);  convert_element_type_default_1 = clamp_max_default = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor, 6);  mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        mean_dim = torch.ops.aten.mean.dim(convert_element_type_default_2, [-1, -2], True, dtype = None);  convert_element_type_default_2 = None
        view_default = torch.ops.aten.view.default(mean_dim, input_32);  mean_dim = input_32 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_34, torch.bfloat16);  input_34 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default_3 = view_default = permute_default = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(addmm_default, torch.float32)
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default_5, 3, alpha = 1)
        clamp_min_default_1 = torch.ops.aten.clamp_min.default(add_tensor_2, 0);  add_tensor_2 = None
        clamp_max_default_1 = torch.ops.aten.clamp_max.default(clamp_min_default_1, 6);  clamp_min_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, clamp_max_default_1);  convert_element_type_default_5 = clamp_max_default_1 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(mul_tensor_1, 6);  mul_tensor_1 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None
        lt_scalar = torch.ops.aten.lt.Scalar(input_35, 0.8);  input_35 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(lt_scalar, torch.bfloat16);  lt_scalar = None
        div_scalar = torch.ops.aten.div.Scalar(convert_element_type_default_7, 0.8);  convert_element_type_default_7 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, div_scalar);  convert_element_type_default_6 = div_scalar = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_36, torch.bfloat16);  input_36 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_37, torch.bfloat16);  input_37 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_9, [1, 0]);  convert_element_type_default_9 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_8, mul_tensor_2, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_8 = permute_default_1 = None
        return (addmm_default, mul_tensor_2, addmm_default_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
