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
    def forward(self, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67):
        add_tensor = torch.ops.aten.add.Tensor(input_46, input_47, alpha = 1);  input_46 = input_47 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_48, 1.1111111111111112);  input_48 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_49, 0.1);  input_49 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor_1);  gt_scalar = add_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None
        expand_default = torch.ops.aten.expand.default(input_50, input_51, implicit = False);  input_50 = input_51 = None
        view_default = torch.ops.aten.view.default(expand_default, input_52);  expand_default = input_52 = None
        clone_default = torch.ops.aten.clone.default(input_53, memory_format = torch.contiguous_format);  input_53 = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_54);  clone_default = input_54 = None
        bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default, input_55);  bmm_default = input_55 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default_1, input_56);  clone_default_1 = input_56 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_57, torch.bfloat16);  input_57 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_59);  view_default_3 = input_59 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_4, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default, input_60);  addmm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_61, torch.bfloat16);  input_61 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_2, 0.1);  convert_element_type_default_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default_5);  gt_scalar_1 = view_default_5 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_4, alpha = 1);  mul_tensor_2 = mul_tensor_4 = None
        mean_dim = torch.ops.aten.mean.dim(add_tensor_2, [-1], True, dtype = None)
        var_correction = torch.ops.aten.var.correction(add_tensor_2, [-1], correction = 1.0, keepdim = True)
        sqrt_default = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, mean_dim, alpha = 1);  mean_dim = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_62, sub_tensor);  input_62 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(sqrt_default, 1e-06, alpha = 1)
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor_5, add_tensor_3);  mul_tensor_5 = add_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(div_tensor, input_63, alpha = 1);  div_tensor = input_63 = None
        view_default_6 = torch.ops.aten.view.default(input_64, input_60);  input_64 = input_60 = None
        gt_scalar_2 = torch.ops.aten.gt.Scalar(input_65, 0.1);  input_65 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(gt_scalar_2, view_default_6);  gt_scalar_2 = view_default_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.1111111111111112);  mul_tensor_6 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_7, alpha = 1);  add_tensor_2 = mul_tensor_7 = None
        gt_scalar_3 = torch.ops.aten.gt.Scalar(input_66, 0.1);  input_66 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(gt_scalar_3, add_tensor_5);  gt_scalar_3 = add_tensor_5 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 1.1111111111111112);  mul_tensor_8 = None
        mean_dim_1 = torch.ops.aten.mean.dim(mul_tensor_9, [-1], True, dtype = None)
        var_correction_1 = torch.ops.aten.var.correction(mul_tensor_9, [-1], correction = 1.0, keepdim = True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_9, mean_dim_1, alpha = 1);  mean_dim_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_9, input_67, alpha = 1);  mul_tensor_9 = input_67 = None
        return (view_default_4, sqrt_default, sub_tensor, add_tensor_4, var_correction_1, sub_tensor_1, add_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
