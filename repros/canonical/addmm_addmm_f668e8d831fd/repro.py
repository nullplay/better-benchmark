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
    def forward(self, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52):
        mul_tensor = torch.ops.aten.mul.Tensor(input_37, input_38);  input_37 = input_38 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_39, mul_tensor_1, alpha = 1);  input_39 = mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(input_40, input_41);  input_40 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_42, 0.1);  input_42 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_3, alpha = 1);  add_tensor = mul_tensor_3 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_43);  mul_tensor_4 = input_43 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_5, input_44, alpha = 1);  mul_tensor_5 = input_44 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_45, torch.bfloat16);  input_45 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_46, torch.bfloat16);  input_46 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_2, input_47);  convert_element_type_default_2 = input_47 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_1, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_48);  addmm_default = input_48 = None
        relu_default = torch.ops.aten.relu.default(view_default_2);  view_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_49, torch.bfloat16);  input_49 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        view_default_3 = torch.ops.aten.view.default(relu_default, input_51);  input_51 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_41);  addmm_default_1 = input_41 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_52, torch.bfloat16);  input_52 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_5, 0.1);  convert_element_type_default_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default_4);  gt_scalar_1 = view_default_4 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.1111111111111112);  mul_tensor_6 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_7, alpha = 1);  add_tensor_1 = mul_tensor_7 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (view_default_3, add_tensor_4, le_scalar)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
