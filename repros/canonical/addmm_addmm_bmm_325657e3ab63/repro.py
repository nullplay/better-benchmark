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
    def forward(self, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154, input_155, input_156, input_157, input_158, input_159, input_160, input_161, input_162, input_163, input_164, input_165, input_166, input_167, input_168, input_169, input_170, input_171, input_172, input_173, input_174, input_175, input_176, input_177, input_178, input_179, input_180, input_181, input_182, input_183):
        embedding_default = torch.ops.aten.embedding.default(input_136, input_137, -1, False, False);  input_136 = input_137 = None
        add_tensor = torch.ops.aten.add.Tensor(input_138, 0, alpha = 1);  input_138 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_139, unsqueeze_default, -1, False, False);  input_139 = unsqueeze_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1, alpha = 1);  embedding_default = embedding_default_1 = None
        view_default = torch.ops.aten.view.default(input_140, input_141);  input_140 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_142, torch.bfloat16);  input_142 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_143, torch.bfloat16);  input_143 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_144);  view_default = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_1, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_141);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_2, add_tensor_1, alpha = 1);  view_default_2 = add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_3 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_145);  input_145 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_1, input_146, alpha = 1);  mul_tensor_1 = input_146 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_147, input_148);  input_147 = input_148 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_149, torch.bfloat16);  input_149 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_150, torch.bfloat16);  input_150 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_4, input_151);  convert_element_type_default_4 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_2, view_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_2 = view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_141);  addmm_default_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_4, alpha = 1);  add_tensor_2 = view_default_4 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_5, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_152);  input_152 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_4, input_153, alpha = 1);  mul_tensor_4 = input_153 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        expand_default = torch.ops.aten.expand.default(input_154, input_155, implicit = False);  input_154 = None
        view_default_5 = torch.ops.aten.view.default(expand_default, input_156);  expand_default = None
        clone_default = torch.ops.aten.clone.default(input_157, memory_format = torch.contiguous_format);  input_157 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_156);  clone_default = input_156 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_5, view_default_6);  view_default_5 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_155);  bmm_default = input_155 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_1, input_141);  clone_default_1 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_158, torch.bfloat16);  input_158 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_159, torch.bfloat16);  input_159 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_144);  view_default_8 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_9, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_3 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_2, input_141);  addmm_default_2 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_10, add_tensor_5, alpha = 1);  view_default_10 = add_tensor_5 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_8, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_8, getitem_5, alpha = 1);  getitem_5 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_160);  input_160 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor_6, input_161, alpha = 1);  mul_tensor_6 = input_161 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.bfloat16);  add_tensor_10 = None
        view_default_11 = torch.ops.aten.view.default(convert_element_type_default_8, input_144);  convert_element_type_default_8 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_162, input_163);  input_162 = input_163 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_164, torch.bfloat16);  input_164 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_165, torch.bfloat16);  input_165 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_11, input_151);  convert_element_type_default_11 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0]);  convert_element_type_default_10 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_9, view_default_12, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_9 = view_default_12 = permute_default_4 = None
        view_default_13 = torch.ops.aten.view.default(addmm_default_3, input_141);  addmm_default_3 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_8, view_default_13, alpha = 1);  add_tensor_8 = view_default_13 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_11, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(getitem_6, 1e-05, alpha = 1);  getitem_6 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_11, getitem_7, alpha = 1);  getitem_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, input_166);  input_166 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(mul_tensor_9, input_167, alpha = 1);  mul_tensor_9 = input_167 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(add_tensor_13, torch.bfloat16);  add_tensor_13 = None
        view_default_14 = torch.ops.aten.view.default(input_168, input_141);  input_168 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(input_169, torch.bfloat16);  input_169 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(input_170, torch.bfloat16);  input_170 = None
        view_default_15 = torch.ops.aten.view.default(view_default_14, input_144);  view_default_14 = input_144 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_14, [1, 0]);  convert_element_type_default_14 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_13, view_default_15, permute_default_5, beta = 1, alpha = 1);  convert_element_type_default_13 = view_default_15 = permute_default_5 = None
        view_default_16 = torch.ops.aten.view.default(addmm_default_4, input_141);  addmm_default_4 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(view_default_16, add_tensor_11, alpha = 1);  view_default_16 = add_tensor_11 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(add_tensor_14, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_correction_4[0]
        getitem_9 = var_mean_correction_4[1];  var_mean_correction_4 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(getitem_8, 1e-05, alpha = 1);  getitem_8 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_15);  add_tensor_15 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_14, getitem_9, alpha = 1);  getitem_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_4);  sub_tensor_4 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, input_171);  input_171 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(mul_tensor_11, input_172, alpha = 1);  mul_tensor_11 = input_172 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(input_173, torch.bfloat16);  input_173 = None
        view_default_17 = torch.ops.aten.view.default(input_174, input_151);  input_174 = input_151 = None
        permute_default_6 = torch.ops.aten.permute.default(input_175, [1, 0]);  input_175 = None
        addmm_default_5 = torch.ops.aten.addmm.default(convert_element_type_default_15, view_default_17, permute_default_6, beta = 1, alpha = 1);  convert_element_type_default_15 = view_default_17 = permute_default_6 = None
        view_default_18 = torch.ops.aten.view.default(addmm_default_5, input_141);  addmm_default_5 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_14, view_default_18, alpha = 1);  add_tensor_14 = view_default_18 = None
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(add_tensor_17, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_correction_5[0]
        getitem_11 = var_mean_correction_5[1];  var_mean_correction_5 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(getitem_10, 1e-05, alpha = 1);  getitem_10 = None
        rsqrt_default_5 = torch.ops.aten.rsqrt.default(add_tensor_18);  add_tensor_18 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(add_tensor_17, getitem_11, alpha = 1);  getitem_11 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_default_5);  sub_tensor_5 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, input_176);  input_176 = None
        addmm_default_6 = torch.ops.aten.addmm.default(input_177, input_178, input_179, beta = 1, alpha = 1);  input_177 = input_178 = input_179 = None
        view_default_19 = torch.ops.aten.view.default(addmm_default_6, input_141);  addmm_default_6 = None
        add_tensor_19 = torch.ops.aten.add.Tensor(view_default_19, add_tensor_17, alpha = 1);  view_default_19 = add_tensor_17 = None
        var_mean_correction_6 = torch.ops.aten.var_mean.correction(add_tensor_19, [2], correction = 0, keepdim = True)
        getitem_12 = var_mean_correction_6[0]
        getitem_13 = var_mean_correction_6[1];  var_mean_correction_6 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(add_tensor_19, getitem_13, alpha = 1);  getitem_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(sub_tensor_6, input_180);  sub_tensor_6 = input_180 = None
        view_default_20 = torch.ops.aten.view.default(input_181, input_141);  input_181 = input_141 = None
        add_tensor_20 = torch.ops.aten.add.Tensor(add_tensor_19, view_default_20, alpha = 1);  add_tensor_19 = view_default_20 = None
        var_mean_correction_7 = torch.ops.aten.var_mean.correction(add_tensor_20, [2], correction = 0, keepdim = True)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(add_tensor_20, input_182, alpha = 1);  input_182 = None
        add_tensor_21 = torch.ops.aten.add.Tensor(input_183, add_tensor_20, alpha = 1);  input_183 = add_tensor_20 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_4, 2048);  rsqrt_default_4 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(rsqrt_default_3, 2048);  rsqrt_default_3 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(rsqrt_default_2, 2048);  rsqrt_default_2 = None
        div_tensor_3 = torch.ops.aten.div.Tensor(rsqrt_default_1, 2048);  rsqrt_default_1 = None
        div_tensor_4 = torch.ops.aten.div.Tensor(rsqrt_default, 2048);  rsqrt_default = None
        return (mul_tensor, add_tensor_4, mul_tensor_3, convert_element_type_default_5, view_default_9, mul_tensor_5, view_default_11, mul_tensor_8, convert_element_type_default_12, mul_tensor_10, add_tensor_16, rsqrt_default_5, mul_tensor_12, mul_tensor_13, getitem_12, mul_tensor_14, var_mean_correction_7, sub_tensor_7, add_tensor_21, div_tensor, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
