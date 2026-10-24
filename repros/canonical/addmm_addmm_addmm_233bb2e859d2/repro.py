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
    def forward(self, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154, input_155, input_156, input_157, input_158, input_159, input_160, input_161, input_162, input_163, input_164):
        convolution_default = torch.ops.aten.convolution.default(input_122, input_123, input_124, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  input_122 = input_123 = input_124 = None
        view_default = torch.ops.aten.view.default(convolution_default, input_125);  convolution_default = input_125 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
        expand_default = torch.ops.aten.expand.default(input_126, input_127, implicit = False);  input_126 = input_127 = None
        cat_default = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None
        add_tensor = torch.ops.aten.add.Tensor(cat_default, input_128, alpha = 1);  cat_default = input_128 = None
        permute_default_1 = torch.ops.aten.permute.default(input_129, [0, 2, 1, 3]);  input_129 = None
        view_default_1 = torch.ops.aten.view.default(permute_default_1, input_130);  permute_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_131, torch.bfloat16);  input_131 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_132, torch.bfloat16);  input_132 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_133);  view_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_2, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default = view_default_2 = permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default, input_130);  addmm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_3, alpha = 1);  add_tensor = view_default_3 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-06, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_134);  input_134 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_135, alpha = 1);  mul_tensor_1 = input_135 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_136, torch.bfloat16);  input_136 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_3, input_133);  convert_element_type_default_3 = None
        permute_default_3 = torch.ops.aten.permute.default(input_137, [1, 0]);  input_137 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_2, view_default_4, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_2 = permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_1, input_138);  input_138 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.5)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.7071067811865476);  convert_element_type_default_4 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_4);  mul_tensor_2 = add_tensor_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_139, torch.bfloat16);  input_139 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_140, torch.bfloat16);  input_140 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_5, input_141);  convert_element_type_default_5 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_6, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_4 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_2, input_130);  addmm_default_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_7, alpha = 1);  add_tensor_1 = view_default_7 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-06, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_5, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_142);  input_142 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_6, input_143, alpha = 1);  mul_tensor_6 = input_143 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_8, input_133);  convert_element_type_default_8 = None
        permute_default_5 = torch.ops.aten.permute.default(input_144, [0, 2, 1, 3]);  input_144 = None
        view_default_9 = torch.ops.aten.view.default(permute_default_5, input_130);  permute_default_5 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_145, torch.bfloat16);  input_145 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_146, torch.bfloat16);  input_146 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_133);  view_default_9 = None
        permute_default_6 = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0]);  convert_element_type_default_10 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_9, view_default_10, permute_default_6, beta = 1, alpha = 1);  convert_element_type_default_9 = view_default_10 = permute_default_6 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_3, input_130);  addmm_default_3 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_11, alpha = 1);  add_tensor_5 = view_default_11 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_8, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem_4, 1e-06, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_8, getitem_5, alpha = 1);  getitem_5 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_147);  input_147 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor_8, input_148, alpha = 1);  mul_tensor_8 = input_148 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.bfloat16);  add_tensor_10 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(input_149, torch.bfloat16);  input_149 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(input_150, torch.bfloat16);  input_150 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(input_151, torch.bfloat16);  input_151 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_12, input_141);  convert_element_type_default_12 = input_141 = None
        permute_default_7 = torch.ops.aten.permute.default(convert_element_type_default_14, [1, 0]);  convert_element_type_default_14 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_13, view_default_12, permute_default_7, beta = 1, alpha = 1);  convert_element_type_default_13 = view_default_12 = permute_default_7 = None
        view_default_13 = torch.ops.aten.view.default(addmm_default_4, input_130);  addmm_default_4 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_8, view_default_13, alpha = 1);  add_tensor_8 = view_default_13 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_11, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(getitem_6, 1e-06, alpha = 1);  getitem_6 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_11, getitem_7, alpha = 1);  getitem_7 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, input_152);  input_152 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(mul_tensor_10, input_153, alpha = 1);  mul_tensor_10 = input_153 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(input_154, torch.bfloat16);  input_154 = None
        view_default_14 = torch.ops.aten.view.default(input_155, input_133);  input_155 = input_133 = None
        permute_default_8 = torch.ops.aten.permute.default(input_156, [1, 0]);  input_156 = None
        addmm_default_5 = torch.ops.aten.addmm.default(convert_element_type_default_15, view_default_14, permute_default_8, beta = 1, alpha = 1);  convert_element_type_default_15 = view_default_14 = permute_default_8 = None
        view_default_15 = torch.ops.aten.view.default(addmm_default_5, input_130);  addmm_default_5 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(add_tensor_11, view_default_15, alpha = 1);  add_tensor_11 = view_default_15 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(add_tensor_14, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_correction_4[0]
        getitem_9 = var_mean_correction_4[1];  var_mean_correction_4 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(getitem_8, 1e-06, alpha = 1);  getitem_8 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_15);  add_tensor_15 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_14, getitem_9, alpha = 1);  getitem_9 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_4);  sub_tensor_4 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, input_157);  input_157 = None
        addmm_default_6 = torch.ops.aten.addmm.default(input_158, input_159, input_160, beta = 1, alpha = 1);  input_158 = input_159 = input_160 = None
        view_default_16 = torch.ops.aten.view.default(addmm_default_6, input_130);  addmm_default_6 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(add_tensor_14, view_default_16, alpha = 1);  add_tensor_14 = view_default_16 = None
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(add_tensor_16, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_correction_5[0]
        getitem_11 = var_mean_correction_5[1];  var_mean_correction_5 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(add_tensor_16, getitem_11, alpha = 1);  getitem_11 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_5, input_161);  sub_tensor_5 = input_161 = None
        view_default_17 = torch.ops.aten.view.default(input_162, input_130);  input_162 = input_130 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_16, view_default_17, alpha = 1);  add_tensor_16 = view_default_17 = None
        var_mean_correction_6 = torch.ops.aten.var_mean.correction(add_tensor_17, [2], correction = 0, keepdim = True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(add_tensor_17, input_163, alpha = 1);  input_163 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(add_tensor_17, input_164, alpha = 1);  add_tensor_17 = input_164 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_3, 192);  rsqrt_default_3 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(rsqrt_default_2, 192);  rsqrt_default_2 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(rsqrt_default_1, 192);  rsqrt_default_1 = None
        div_tensor_3 = torch.ops.aten.div.Tensor(rsqrt_default, 192);  rsqrt_default = None
        return (mul_tensor, view_default_4, addmm_default_1, view_default_6, mul_tensor_5, view_default_8, mul_tensor_7, convert_element_type_default_11, mul_tensor_9, add_tensor_13, rsqrt_default_4, mul_tensor_11, mul_tensor_12, getitem_10, mul_tensor_13, var_mean_correction_6, sub_tensor_6, add_tensor_18, div_tensor, div_tensor_1, div_tensor_2, div_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
