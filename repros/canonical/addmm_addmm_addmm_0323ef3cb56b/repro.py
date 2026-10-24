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
    def forward(self, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135):
        expand_default = torch.ops.aten.expand.default(input_93, input_94, implicit = False);  input_93 = input_94 = None
        permute_default = torch.ops.aten.permute.default(input_95, [0, 2, 1]);  input_95 = None
        cat_default = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None
        view_default = torch.ops.aten.view.default(input_96, input_97);  input_96 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_98, view_default);  input_98 = view_default = None
        add_tensor = torch.ops.aten.add.Tensor(cat_default, mul_tensor, alpha = 1);  cat_default = mul_tensor = None
        addmm_default = torch.ops.aten.addmm.default(input_99, input_100, input_101, beta = 1, alpha = 1);  input_99 = input_100 = input_101 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_97);  addmm_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_102, view_default_1);  input_102 = view_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_1, alpha = 1);  add_tensor = mul_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_103);  sub_tensor = input_103 = None
        view_default_2 = torch.ops.aten.view.default(input_104, input_105);  input_104 = None
        permute_default_1 = torch.ops.aten.permute.default(input_106, [1, 0]);  input_106 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_107, view_default_2, permute_default_1, beta = 1, alpha = 1);  input_107 = view_default_2 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_97);  addmm_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_108, view_default_3);  input_108 = view_default_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_3, alpha = 1);  add_tensor_1 = mul_tensor_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_3, alpha = 1);  convert_element_type_default_1 = getitem_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(getitem_2, 1e-06, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_109);  mul_tensor_4 = input_109 = None
        view_default_4 = torch.ops.aten.view.default(input_110, input_105);  input_110 = None
        permute_default_2 = torch.ops.aten.permute.default(input_111, [1, 0]);  input_111 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_112, view_default_4, permute_default_2, beta = 1, alpha = 1);  input_112 = view_default_4 = permute_default_2 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_113);  addmm_default_2 = input_113 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_7);  mul_tensor_7 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_6, add_tensor_4);  mul_tensor_6 = add_tensor_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_3, input_114);  convert_element_type_default_3 = input_114 = None
        permute_default_3 = torch.ops.aten.permute.default(input_115, [1, 0]);  input_115 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_116, view_default_6, permute_default_3, beta = 1, alpha = 1);  input_116 = view_default_6 = permute_default_3 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_3, input_97);  addmm_default_3 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(input_117, view_default_7);  input_117 = view_default_7 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_9, alpha = 1);  add_tensor_2 = mul_tensor_9 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_5, alpha = 1);  convert_element_type_default_4 = getitem_5 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_4, 1e-06, alpha = 1);  getitem_4 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_1);  sub_tensor_2 = rsqrt_default_1 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, input_118);  mul_tensor_10 = input_118 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_11, input_119, alpha = 1);  mul_tensor_11 = input_119 = None
        view_default_8 = torch.ops.aten.view.default(input_120, input_105);  input_120 = input_105 = None
        permute_default_4 = torch.ops.aten.permute.default(input_121, [1, 0]);  input_121 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_122, view_default_8, permute_default_4, beta = 1, alpha = 1);  input_122 = view_default_8 = permute_default_4 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default_4, input_97);  addmm_default_4 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_123, view_default_9);  input_123 = view_default_9 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_12, alpha = 1);  add_tensor_5 = mul_tensor_12 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, getitem_7, alpha = 1);  convert_element_type_default_5 = getitem_7 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem_6, 1e-06, alpha = 1);  getitem_6 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_2);  sub_tensor_3 = rsqrt_default_2 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, input_124);  mul_tensor_13 = input_124 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_125, input_126, input_127, beta = 1, alpha = 1);  input_125 = input_126 = input_127 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_5, input_97);  addmm_default_5 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(input_128, view_default_10);  input_128 = view_default_10 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_8, mul_tensor_15, alpha = 1);  add_tensor_8 = mul_tensor_15 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.float32)
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convert_element_type_default_6, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_correction_4[0]
        getitem_9 = var_mean_correction_4[1];  var_mean_correction_4 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_6, getitem_9, alpha = 1);  convert_element_type_default_6 = getitem_9 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(sub_tensor_4, input_129);  sub_tensor_4 = input_129 = None
        view_default_11 = torch.ops.aten.view.default(input_130, input_97);  input_130 = input_97 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(input_131, view_default_11);  input_131 = view_default_11 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_10, mul_tensor_17, alpha = 1);  add_tensor_10 = mul_tensor_17 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.float32)
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convert_element_type_default_7, [2], correction = 0, keepdim = True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_7, input_132, alpha = 1);  convert_element_type_default_7 = input_132 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(input_133, input_134);  input_133 = input_134 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_11, mul_tensor_18, alpha = 1);  add_tensor_11 = mul_tensor_18 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_12, torch.float32)
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_12, input_135, alpha = 1);  add_tensor_12 = input_135 = None
        return (getitem, mul_tensor_2, mul_tensor_5, add_tensor_7, mul_tensor_14, getitem_8, mul_tensor_16, var_mean_correction_5, sub_tensor_5, convert_element_type_default_8, add_tensor_13)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
