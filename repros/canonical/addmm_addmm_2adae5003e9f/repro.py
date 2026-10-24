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
    def forward(self, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101):
        add_tensor = torch.ops.aten.add.Tensor(input_74, input_75, alpha = 1);  input_74 = input_75 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_76, 1.1111111111111112);  input_76 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, add_tensor, alpha = 1);  mul_tensor = add_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_77, input_78);  input_77 = input_78 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_2, alpha = 1);  add_tensor_1 = mul_tensor_2 = None
        view_default = torch.ops.aten.view.default(input_79, input_80);  input_79 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_81, 0.1);  input_81 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_4, add_tensor_2, alpha = 1);  mul_tensor_4 = add_tensor_2 = None
        view_default_1 = torch.ops.aten.view.default(input_82, input_83);  input_82 = input_83 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_84, torch.bfloat16);  input_84 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_85, torch.bfloat16);  input_85 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_2, convert_element_type_default_1, convert_element_type_default, beta = 1, alpha = 1);  convert_element_type_default_2 = convert_element_type_default_1 = convert_element_type_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_86);  input_86 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_3, 3.0);  convert_element_type_default_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_2, mul_tensor_6, alpha = 1);  view_default_2 = mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(add_tensor_4, 0.7978845608028654);  add_tensor_4 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_7);  mul_tensor_7 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_5, add_tensor_5);  mul_tensor_5 = add_tensor_5 = None
        view_default_3 = torch.ops.aten.view.default(mul_tensor_8, input_87);  mul_tensor_8 = input_87 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_88, torch.bfloat16);  input_88 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_3, torch.bfloat16);  view_default_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_89, torch.bfloat16);  input_89 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_6, convert_element_type_default_5, convert_element_type_default_4, beta = 1, alpha = 1);  convert_element_type_default_6 = convert_element_type_default_4 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_80);  addmm_default_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_90, torch.bfloat16);  input_90 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.1);  convert_element_type_default_7 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default_4);  gt_scalar_1 = view_default_4 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, 1.1111111111111112);  mul_tensor_9 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_10, alpha = 1);  add_tensor_3 = mul_tensor_10 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_6, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_6, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, input_91);  input_91 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_12, input_92, alpha = 1);  mul_tensor_12 = input_92 = None
        view_default_5 = torch.ops.aten.view.default(input_93, input_80);  input_93 = input_80 = None
        gt_scalar_2 = torch.ops.aten.gt.Scalar(input_94, 0.1);  input_94 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(gt_scalar_2, view_default_5);  gt_scalar_2 = view_default_5 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, 1.1111111111111112);  mul_tensor_13 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(mul_tensor_14, add_tensor_6, alpha = 1);  mul_tensor_14 = add_tensor_6 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_9, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_9, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_15, input_95);  input_95 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(input_96, input_97);  input_96 = input_97 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_17, 1.1111111111111112);  mul_tensor_17 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_9, mul_tensor_18, alpha = 1);  add_tensor_9 = mul_tensor_18 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_11, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_11, getitem_5, alpha = 1);  getitem_5 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_98);  sub_tensor_2 = input_98 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(input_99, 1.1111111111111112);  input_99 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(mul_tensor_20, add_tensor_11, alpha = 1);  mul_tensor_20 = add_tensor_11 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_12, [2], correction = 0, keepdim = True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_12, input_100, alpha = 1);  input_100 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_12, input_101, alpha = 1);  add_tensor_12 = input_101 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default, 768);  rsqrt_default = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        return (addmm_default, mul_tensor_11, add_tensor_8, rsqrt_default_1, mul_tensor_15, mul_tensor_16, getitem_4, mul_tensor_19, var_mean_correction_3, sub_tensor_3, add_tensor_13, div_tensor, permute_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
