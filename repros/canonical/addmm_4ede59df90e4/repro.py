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
    def forward(self, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53):
        getitem = input_39[0];  input_39 = None
        permute_default = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default = torch.ops.aten.view.default(permute_default, input_40);  permute_default = input_40 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_43);  view_default = input_43 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_1, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default = view_default_1 = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_44)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_45, torch.bfloat16);  input_45 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, view_default_2);  gt_scalar = view_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_46, mul_tensor_1, alpha = 1);  input_46 = mul_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem_1 = var_mean_correction[0]
        getitem_2 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem_1, 1e-05, alpha = 1);  getitem_1 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, getitem_2, alpha = 1)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_47);  mul_tensor_2 = input_47 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_3, input_48, alpha = 1);  mul_tensor_3 = input_48 = None
        view_default_3 = torch.ops.aten.view.default(input_49, input_44);  input_49 = input_44 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(input_50, 0.1);  input_50 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default_3);  gt_scalar_1 = view_default_3 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_5, alpha = 1);  add_tensor = mul_tensor_5 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_3, [2], correction = 0, keepdim = True)
        getitem_3 = var_mean_correction_1[0]
        getitem_4 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem_3, 1e-05, alpha = 1);  getitem_3 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_4, alpha = 1);  getitem_4 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_51);  input_51 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_52, input_53);  input_52 = input_53 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 1.1111111111111112);  mul_tensor_8 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_9, alpha = 1);  add_tensor_3 = mul_tensor_9 = None
        return (addmm_default, getitem_2, rsqrt_default, add_tensor_2, rsqrt_default_1, mul_tensor_6, mul_tensor_7, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
