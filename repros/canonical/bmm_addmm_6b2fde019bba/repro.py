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
    def forward(self, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50):
        expand_default = torch.ops.aten.expand.default(input_34, input_35, implicit = False);  input_34 = input_35 = None
        view_default = torch.ops.aten.view.default(expand_default, input_36);  expand_default = input_36 = None
        clone_default = torch.ops.aten.clone.default(input_37, memory_format = torch.contiguous_format);  input_37 = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_38);  clone_default = input_38 = None
        bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default, input_39);  bmm_default = input_39 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default_1, input_40);  clone_default_1 = input_40 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_43);  view_default_3 = input_43 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_4, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default, input_44)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_45, torch.bfloat16);  input_45 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, view_default_5);  gt_scalar = view_default_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_46, mul_tensor_1, alpha = 1);  input_46 = mul_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, getitem_1, alpha = 1)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_47);  mul_tensor_2 = input_47 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_3, input_48, alpha = 1);  mul_tensor_3 = input_48 = None
        view_default_6 = torch.ops.aten.view.default(input_49, input_44);  input_49 = input_44 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(input_50, 0.1);  input_50 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default_6);  gt_scalar_1 = view_default_6 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_5, alpha = 1);  add_tensor = mul_tensor_5 = None
        return (view_default_4, addmm_default, getitem_1, rsqrt_default, add_tensor_2, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
