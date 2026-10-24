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
    def forward(self, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154, input_155, input_156, input_157, input_158, input_159, input_160, input_161, input_162, input_163, input_164, input_165, input_166, input_167, input_168, input_169, input_170, input_171, input_172, input_173, input_174, input_175, input_176, input_177, input_178, input_179, input_180, input_181):
        add_tensor = torch.ops.aten.add.Tensor(input_137, input_138, alpha = 1);  input_137 = input_138 = None
        view_default = torch.ops.aten.view.default(input_139, input_140);  input_139 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        addmm_default = torch.ops.aten.addmm.default(input_141, input_142, input_143, beta = 1, alpha = 1);  input_141 = input_142 = input_143 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_140);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, input_144);  sub_tensor = input_144 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_145, torch.bfloat16);  input_145 = None
        view_default_2 = torch.ops.aten.view.default(input_146, input_147);  input_146 = None
        permute_default = torch.ops.aten.permute.default(input_148, [1, 0]);  input_148 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default, view_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_140);  addmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_3, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-06, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_149);  input_149 = None
        view_default_4 = torch.ops.aten.view.default(input_150, input_140);  input_150 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_151, torch.bfloat16);  input_151 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_152, torch.bfloat16);  input_152 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_153);  view_default_4 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_5, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_5 = permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_140);  addmm_default_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_6, alpha = 1);  add_tensor_3 = view_default_6 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_5, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_4, 1e-06, alpha = 1);  getitem_4 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_5, getitem_5, alpha = 1);  getitem_5 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_1);  sub_tensor_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_154);  input_154 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_4, input_155, alpha = 1);  mul_tensor_4 = input_155 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_156, input_157);  input_156 = input_157 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_158, torch.bfloat16);  input_158 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_159, torch.bfloat16);  input_159 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_3, input_147);  convert_element_type_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_7, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_4 = view_default_7 = permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_140);  addmm_default_3 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_8, alpha = 1);  add_tensor_5 = view_default_8 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_8, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem_6, 1e-06, alpha = 1);  getitem_6 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_8, getitem_7, alpha = 1);  getitem_7 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_2);  sub_tensor_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_160);  input_160 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor_7, input_161, alpha = 1);  mul_tensor_7 = input_161 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.bfloat16);  add_tensor_10 = None
        getitem_8 = input_162[0];  input_162 = None
        permute_default_3 = torch.ops.aten.permute.default(getitem_8, [0, 2, 1, 3]);  getitem_8 = None
        view_default_9 = torch.ops.aten.view.default(permute_default_3, input_140);  permute_default_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_163, torch.bfloat16);  input_163 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_164, torch.bfloat16);  input_164 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_153);  view_default_9 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_8, [1, 0]);  convert_element_type_default_8 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_7, view_default_10, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_7 = view_default_10 = permute_default_4 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_4, input_140);  addmm_default_4 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_8, view_default_11, alpha = 1);  add_tensor_8 = view_default_11 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(add_tensor_11, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_correction_4[0]
        getitem_10 = var_mean_correction_4[1];  var_mean_correction_4 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(getitem_9, 1e-06, alpha = 1);  getitem_9 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_11, getitem_10, alpha = 1);  getitem_10 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_3);  sub_tensor_4 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, input_165);  input_165 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(mul_tensor_9, input_166, alpha = 1);  mul_tensor_9 = input_166 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor_13, torch.bfloat16);  add_tensor_13 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_9, input_153);  convert_element_type_default_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_167, input_168);  input_167 = input_168 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(input_169, torch.bfloat16);  input_169 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(input_170, torch.bfloat16);  input_170 = None
        view_default_13 = torch.ops.aten.view.default(convert_element_type_default_10, input_147);  convert_element_type_default_10 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_12, [1, 0]);  convert_element_type_default_12 = None
        addmm_default_5 = torch.ops.aten.addmm.default(convert_element_type_default_11, view_default_13, permute_default_5, beta = 1, alpha = 1);  convert_element_type_default_11 = view_default_13 = permute_default_5 = None
        view_default_14 = torch.ops.aten.view.default(addmm_default_5, input_140);  addmm_default_5 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(add_tensor_11, view_default_14, alpha = 1);  add_tensor_11 = view_default_14 = None
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(add_tensor_14, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_correction_5[0]
        getitem_12 = var_mean_correction_5[1];  var_mean_correction_5 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(getitem_11, 1e-06, alpha = 1);  getitem_11 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_15);  add_tensor_15 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(add_tensor_14, getitem_12, alpha = 1);  getitem_12 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_default_4);  sub_tensor_5 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, input_171);  input_171 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(mul_tensor_12, input_172, alpha = 1);  mul_tensor_12 = input_172 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(add_tensor_16, torch.bfloat16);  add_tensor_16 = None
        view_default_15 = torch.ops.aten.view.default(input_173, input_140);  input_173 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(input_174, torch.bfloat16);  input_174 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(input_175, torch.bfloat16);  input_175 = None
        view_default_16 = torch.ops.aten.view.default(view_default_15, input_153);  view_default_15 = input_153 = None
        permute_default_6 = torch.ops.aten.permute.default(convert_element_type_default_15, [1, 0]);  convert_element_type_default_15 = None
        addmm_default_6 = torch.ops.aten.addmm.default(convert_element_type_default_14, view_default_16, permute_default_6, beta = 1, alpha = 1);  convert_element_type_default_14 = view_default_16 = permute_default_6 = None
        view_default_17 = torch.ops.aten.view.default(addmm_default_6, input_140);  addmm_default_6 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_14, view_default_17, alpha = 1);  add_tensor_14 = view_default_17 = None
        var_mean_correction_6 = torch.ops.aten.var_mean.correction(add_tensor_17, [2], correction = 0, keepdim = True)
        getitem_13 = var_mean_correction_6[0]
        getitem_14 = var_mean_correction_6[1];  var_mean_correction_6 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(getitem_13, 1e-06, alpha = 1);  getitem_13 = None
        rsqrt_default_5 = torch.ops.aten.rsqrt.default(add_tensor_18);  add_tensor_18 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(add_tensor_17, getitem_14, alpha = 1);  getitem_14 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_6, rsqrt_default_5);  sub_tensor_6 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, input_176);  input_176 = None
        add_tensor_19 = torch.ops.aten.add.Tensor(mul_tensor_14, input_177, alpha = 1);  mul_tensor_14 = input_177 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(input_178, torch.bfloat16);  input_178 = None
        view_default_18 = torch.ops.aten.view.default(input_179, input_147);  input_179 = input_147 = None
        permute_default_7 = torch.ops.aten.permute.default(input_180, [1, 0]);  input_180 = None
        addmm_default_7 = torch.ops.aten.addmm.default(convert_element_type_default_16, view_default_18, permute_default_7, beta = 1, alpha = 1);  convert_element_type_default_16 = view_default_18 = permute_default_7 = None
        view_default_19 = torch.ops.aten.view.default(addmm_default_7, input_140);  addmm_default_7 = input_140 = None
        add_tensor_20 = torch.ops.aten.add.Tensor(add_tensor_17, view_default_19, alpha = 1);  add_tensor_17 = view_default_19 = None
        var_mean_correction_7 = torch.ops.aten.var_mean.correction(add_tensor_20, [2], correction = 0, keepdim = True)
        getitem_15 = var_mean_correction_7[0]
        getitem_16 = var_mean_correction_7[1];  var_mean_correction_7 = None
        add_tensor_21 = torch.ops.aten.add.Tensor(getitem_15, 1e-06, alpha = 1);  getitem_15 = None
        rsqrt_default_6 = torch.ops.aten.rsqrt.default(add_tensor_21);  add_tensor_21 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(add_tensor_20, getitem_16, alpha = 1);  add_tensor_20 = getitem_16 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_7, rsqrt_default_6);  sub_tensor_7 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_15, input_181);  input_181 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_5, 768);  rsqrt_default_5 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(rsqrt_default_4, 768);  rsqrt_default_4 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(rsqrt_default_3, 768);  rsqrt_default_3 = None
        div_tensor_3 = torch.ops.aten.div.Tensor(rsqrt_default_2, 768);  rsqrt_default_2 = None
        div_tensor_4 = torch.ops.aten.div.Tensor(rsqrt_default_1, 768);  rsqrt_default_1 = None
        return (getitem, mul_tensor, rsqrt_default, mul_tensor_1, mul_tensor_2, mul_tensor_3, add_tensor_7, mul_tensor_6, convert_element_type_default_6, mul_tensor_8, view_default_12, mul_tensor_11, convert_element_type_default_13, mul_tensor_13, add_tensor_19, rsqrt_default_6, mul_tensor_15, mul_tensor_16, div_tensor, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
