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
    def forward(self, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154, input_155, input_156, input_157, input_158, input_159, input_160, input_161, input_162, input_163, input_164, input_165):
        embedding_default = torch.ops.aten.embedding.default(input_120, input_121, 0, False, False);  input_120 = input_121 = None
        mul_tensor = torch.ops.aten.mul.Tensor(embedding_default, 1.0);  embedding_default = None
        add_tensor = torch.ops.aten.add.Tensor(input_122, 0, alpha = 1);  input_122 = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_123, add_tensor, -1, False, False);  input_123 = add_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, embedding_default_1, alpha = 1);  mul_tensor = embedding_default_1 = None
        view_default = torch.ops.aten.view.default(input_124, input_125);  input_124 = None
        permute_default = torch.ops.aten.permute.default(input_126, [1, 0]);  input_126 = None
        addmm_default = torch.ops.aten.addmm.default(input_127, view_default, permute_default, beta = 1, alpha = 1);  input_127 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_128);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, input_129);  sub_tensor = input_129 = None
        view_default_2 = torch.ops.aten.view.default(input_130, input_131);  input_130 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_125);  view_default_2 = None
        permute_default_1 = torch.ops.aten.permute.default(input_132, [1, 0]);  input_132 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_133, view_default_3, permute_default_1, beta = 1, alpha = 1);  input_133 = view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_128);  addmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_4, alpha = 1);  add_tensor_2 = view_default_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_3, alpha = 1);  convert_element_type_default_1 = getitem_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_134);  mul_tensor_2 = input_134 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_135, input_136);  input_135 = input_136 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_2, input_137);  convert_element_type_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(input_138, [1, 0]);  input_138 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_139, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_139 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_128);  addmm_default_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_6, alpha = 1);  add_tensor_3 = view_default_6 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_5, alpha = 1);  convert_element_type_default_3 = getitem_5 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_1);  sub_tensor_2 = rsqrt_default_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_140);  mul_tensor_5 = input_140 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_6, input_141, alpha = 1);  mul_tensor_6 = input_141 = None
        getitem_6 = input_142[0];  input_142 = None
        permute_default_3 = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
        view_default_7 = torch.ops.aten.view.default(permute_default_3, input_131);  permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_125);  view_default_7 = None
        permute_default_4 = torch.ops.aten.permute.default(input_143, [1, 0]);  input_143 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_144, view_default_8, permute_default_4, beta = 1, alpha = 1);  input_144 = view_default_8 = permute_default_4 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default_3, input_128);  addmm_default_3 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_9, alpha = 1);  add_tensor_5 = view_default_9 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [2], correction = 0, keepdim = True)
        getitem_7 = var_mean_correction_3[0]
        getitem_8 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_8, alpha = 1);  convert_element_type_default_4 = getitem_8 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem_7, 1e-05, alpha = 1);  getitem_7 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_2);  sub_tensor_3 = rsqrt_default_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_145);  mul_tensor_7 = input_145 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor_8, input_146, alpha = 1);  mul_tensor_8 = input_146 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.bfloat16);  add_tensor_10 = None
        clone_default = torch.ops.aten.clone.default(input_147, memory_format = torch.contiguous_format);  input_147 = None
        view_default_10 = torch.ops.aten.view.default(clone_default, input_131);  clone_default = input_131 = None
        view_default_11 = torch.ops.aten.view.default(view_default_10, input_125);  view_default_10 = None
        permute_default_5 = torch.ops.aten.permute.default(input_148, [1, 0]);  input_148 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_149, view_default_11, permute_default_5, beta = 1, alpha = 1);  input_149 = view_default_11 = permute_default_5 = None
        view_default_12 = torch.ops.aten.view.default(addmm_default_4, input_128);  addmm_default_4 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_8, view_default_12, alpha = 1);  add_tensor_8 = view_default_12 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.float32)
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convert_element_type_default_6, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_correction_4[0]
        getitem_10 = var_mean_correction_4[1];  var_mean_correction_4 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_6, getitem_10, alpha = 1);  convert_element_type_default_6 = getitem_10 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(getitem_9, 1e-05, alpha = 1);  getitem_9 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_3);  sub_tensor_4 = rsqrt_default_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, input_150);  mul_tensor_9 = input_150 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(mul_tensor_10, input_151, alpha = 1);  mul_tensor_10 = input_151 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_152, torch.bfloat16);  input_152 = None
        view_default_13 = torch.ops.aten.view.default(convert_element_type_default_7, input_137);  convert_element_type_default_7 = input_137 = None
        permute_default_6 = torch.ops.aten.permute.default(input_153, [1, 0]);  input_153 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_154, view_default_13, permute_default_6, beta = 1, alpha = 1);  input_154 = view_default_13 = permute_default_6 = None
        view_default_14 = torch.ops.aten.view.default(addmm_default_5, input_128);  addmm_default_5 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(add_tensor_11, view_default_14, alpha = 1);  add_tensor_11 = view_default_14 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_14, torch.float32)
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convert_element_type_default_8, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_correction_5[0]
        getitem_12 = var_mean_correction_5[1];  var_mean_correction_5 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_8, getitem_12, alpha = 1);  convert_element_type_default_8 = getitem_12 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(getitem_11, 1e-05, alpha = 1);  getitem_11 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_15);  add_tensor_15 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_default_4);  sub_tensor_5 = rsqrt_default_4 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, input_155);  mul_tensor_11 = input_155 = None
        view_default_15 = torch.ops.aten.view.default(input_156, input_125);  input_156 = input_125 = None
        permute_default_7 = torch.ops.aten.permute.default(input_157, [1, 0]);  input_157 = None
        addmm_default_6 = torch.ops.aten.addmm.default(input_158, view_default_15, permute_default_7, beta = 1, alpha = 1);  input_158 = view_default_15 = permute_default_7 = None
        view_default_16 = torch.ops.aten.view.default(addmm_default_6, input_128);  addmm_default_6 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(add_tensor_14, view_default_16, alpha = 1);  add_tensor_14 = view_default_16 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor_16, torch.float32)
        var_mean_correction_6 = torch.ops.aten.var_mean.correction(convert_element_type_default_9, [2], correction = 0, keepdim = True)
        getitem_13 = var_mean_correction_6[0]
        getitem_14 = var_mean_correction_6[1];  var_mean_correction_6 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(convert_element_type_default_9, getitem_14, alpha = 1);  convert_element_type_default_9 = getitem_14 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(sub_tensor_6, input_159);  sub_tensor_6 = input_159 = None
        addmm_default_7 = torch.ops.aten.addmm.default(input_160, input_161, input_162, beta = 1, alpha = 1);  input_160 = input_161 = input_162 = None
        view_default_17 = torch.ops.aten.view.default(addmm_default_7, input_128);  addmm_default_7 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_16, view_default_17, alpha = 1);  add_tensor_16 = view_default_17 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(add_tensor_17, torch.float32)
        var_mean_correction_7 = torch.ops.aten.var_mean.correction(convert_element_type_default_10, [2], correction = 0, keepdim = True)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(convert_element_type_default_10, input_163, alpha = 1);  convert_element_type_default_10 = input_163 = None
        view_default_18 = torch.ops.aten.view.default(input_164, input_128);  input_164 = input_128 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(add_tensor_17, view_default_18, alpha = 1);  add_tensor_17 = view_default_18 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(add_tensor_18, torch.float32)
        add_tensor_19 = torch.ops.aten.add.Tensor(add_tensor_18, input_165, alpha = 1);  add_tensor_18 = input_165 = None
        return (getitem, mul_tensor_1, mul_tensor_3, add_tensor_7, convert_element_type_default_5, add_tensor_13, mul_tensor_12, getitem_13, mul_tensor_13, var_mean_correction_7, sub_tensor_7, convert_element_type_default_11, add_tensor_19)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
