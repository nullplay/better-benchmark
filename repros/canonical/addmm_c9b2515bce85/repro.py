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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39):
        mul_tensor = torch.ops.aten.mul.Tensor(input_30, 1.1111111111111112);  input_30 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_31, 1.1111111111111112);  input_31 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, mul_tensor_1, alpha = 1);  mul_tensor = mul_tensor_1 = None
        mean_dim = torch.ops.aten.mean.dim(add_tensor, [-1], True, dtype = None)
        var_correction = torch.ops.aten.var.correction(add_tensor, [-1], correction = 1.0, keepdim = True)
        sqrt_default = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, mean_dim, alpha = 1);  add_tensor = mean_dim = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_32, sub_tensor);  input_32 = sub_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(sqrt_default, 1e-06, alpha = 1);  sqrt_default = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor_2, add_tensor_1);  mul_tensor_2 = add_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(div_tensor, input_33, alpha = 1);  div_tensor = input_33 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_34, torch.bfloat16);  input_34 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_35, torch.bfloat16);  input_35 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_36);  convert_element_type_default_2 = input_36 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_37);  input_37 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 0.5)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 0.7071067811865476);  convert_element_type_default_3 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_3);  mul_tensor_3 = add_tensor_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_5, 0.1);  convert_element_type_default_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(gt_scalar, convert_element_type_default_4);  gt_scalar = convert_element_type_default_4 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.1111111111111112);  mul_tensor_6 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_7, input_39);  mul_tensor_7 = input_39 = None
        return (addmm_default, view_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
