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
    def forward(self, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45):
        mul_tensor = torch.ops.aten.mul.Tensor(input_31, input_32);  input_31 = input_32 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_33, mul_tensor_1, alpha = 1);  input_33 = mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(input_34, input_35);  input_34 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_36, 0.1);  input_36 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_3, alpha = 1);  add_tensor = mul_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_37, torch.bfloat16);  input_37 = None
        view_default_1 = torch.ops.aten.view.default(input_38, input_39);  input_38 = input_39 = None
        permute_default = torch.ops.aten.permute.default(input_40, [1, 0]);  input_40 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_1, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_41);  input_41 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.5)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.7071067811865476);  convert_element_type_default_1 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_2);  mul_tensor_4 = add_tensor_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_43, torch.bfloat16);  input_43 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_2, input_44);  convert_element_type_default_2 = input_44 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_35);  addmm_default_1 = input_35 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_45, torch.bfloat16);  input_45 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_5, 0.1);  convert_element_type_default_5 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default_4);  gt_scalar_1 = view_default_4 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 1.1111111111111112);  mul_tensor_7 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_8, alpha = 1);  add_tensor_1 = mul_tensor_8 = None
        return (addmm_default, view_default_3, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
