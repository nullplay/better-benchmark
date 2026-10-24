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
    def forward(self, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38):
        getitem = input_27[0];  input_27 = None
        permute_default = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default = torch.ops.aten.view.default(permute_default, input_28);  permute_default = input_28 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_29, torch.bfloat16);  input_29 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_30, torch.bfloat16);  input_30 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_31);  view_default = input_31 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_1, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default = view_default_1 = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_32)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, view_default_2);  gt_scalar = view_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_34, mul_tensor_1, alpha = 1);  input_34 = mul_tensor_1 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor, input_35);  add_tensor = input_35 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default_3, [1], correction = 0, keepdim = True)
        getitem_1 = var_mean_correction[0]
        getitem_2 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem_1, 1e-05, alpha = 1);  getitem_1 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(view_default_3, getitem_2, alpha = 1)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_36);  mul_tensor_2 = input_36 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_37, input_38);  input_37 = input_38 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_3, mul_tensor_5, alpha = 1);  view_default_3 = mul_tensor_5 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_2, input_32);  add_tensor_2 = input_32 = None
        return (addmm_default, getitem_2, rsqrt_default, mul_tensor_3, view_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
