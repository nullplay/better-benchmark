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
    def forward(self, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154, input_155, input_156, input_157, input_158, input_159, input_160, input_161, input_162, input_163, input_164, input_165):
        clone_default = torch.ops.aten.clone.default(input_125, memory_format = torch.contiguous_format);  input_125 = None
        view_default = torch.ops.aten.view.default(clone_default, input_126);  clone_default = None
        addmm_default = torch.ops.aten.addmm.default(input_127, input_128, input_129, beta = 1, alpha = 1);  input_127 = input_128 = input_129 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_126);  addmm_default = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, view_default_1, alpha = 1);  view_default = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_130, alpha = 1);  convert_element_type_default = input_130 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_131, torch.bfloat16);  input_131 = None
        view_default_2 = torch.ops.aten.view.default(input_132, input_133);  input_132 = None
        permute_default = torch.ops.aten.permute.default(input_134, [1, 0]);  input_134 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_126);  addmm_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_3, alpha = 1);  view_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction_1[0]
        getitem_1 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1, alpha = 1);  convert_element_type_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor_1, input_135);  sub_tensor_1 = input_135 = None
        view_default_4 = torch.ops.aten.view.default(input_136, input_126);  input_136 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_137, torch.bfloat16);  input_137 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_138, torch.bfloat16);  input_138 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_139);  view_default_4 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_5, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = view_default_5 = permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_126);  addmm_default_2 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_6, alpha = 1);  view_default_6 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_2[0]
        getitem_3 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, getitem_3, alpha = 1);  convert_element_type_default_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_140);  mul_tensor_1 = input_140 = None
        div_tensor = torch.ops.aten.div.Tensor(input_141, input_142);  input_141 = input_142 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_143, torch.bfloat16);  input_143 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_144, torch.bfloat16);  input_144 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_6, input_133);  convert_element_type_default_6 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_8, [1, 0]);  convert_element_type_default_8 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_7, view_default_7, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_7 = view_default_7 = permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_126);  addmm_default_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_8, alpha = 1);  view_default_8 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_3[0]
        getitem_5 = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_9, getitem_5, alpha = 1);  convert_element_type_default_9 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_1);  sub_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_145);  mul_tensor_3 = input_145 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_4, input_146, alpha = 1);  mul_tensor_4 = input_146 = None
        getitem_6 = input_147[0];  input_147 = None
        permute_default_3 = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
        view_default_9 = torch.ops.aten.view.default(permute_default_3, input_126);  permute_default_3 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_148, torch.bfloat16);  input_148 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(input_149, torch.bfloat16);  input_149 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_139);  view_default_9 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_11, [1, 0]);  convert_element_type_default_11 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_10, view_default_10, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_10 = view_default_10 = permute_default_4 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_4, input_126);  addmm_default_4 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_4, view_default_11, alpha = 1);  view_default_11 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.float32)
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convert_element_type_default_12, [2], correction = 0, keepdim = True)
        getitem_7 = var_mean_correction_4[0]
        getitem_8 = var_mean_correction_4[1];  var_mean_correction_4 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(getitem_7, 1e-05, alpha = 1);  getitem_7 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_12, getitem_8, alpha = 1);  convert_element_type_default_12 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_2);  sub_tensor_4 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_150);  mul_tensor_5 = input_150 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(mul_tensor_6, input_151, alpha = 1);  mul_tensor_6 = input_151 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(add_tensor_9, torch.bfloat16);  add_tensor_9 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(input_152, input_153);  input_152 = input_153 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(input_154, torch.bfloat16);  input_154 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(input_155, torch.bfloat16);  input_155 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_14, input_133);  convert_element_type_default_14 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_16, [1, 0]);  convert_element_type_default_16 = None
        addmm_default_5 = torch.ops.aten.addmm.default(convert_element_type_default_15, view_default_12, permute_default_5, beta = 1, alpha = 1);  convert_element_type_default_15 = view_default_12 = permute_default_5 = None
        view_default_13 = torch.ops.aten.view.default(addmm_default_5, input_126);  addmm_default_5 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_7, view_default_13, alpha = 1);  view_default_13 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.float32)
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convert_element_type_default_17, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_correction_5[0]
        getitem_10 = var_mean_correction_5[1];  var_mean_correction_5 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(getitem_9, 1e-05, alpha = 1);  getitem_9 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_11);  add_tensor_11 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_17, getitem_10, alpha = 1);  convert_element_type_default_17 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_default_3);  sub_tensor_5 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_156);  mul_tensor_7 = input_156 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(mul_tensor_8, input_157, alpha = 1);  mul_tensor_8 = input_157 = None
        view_default_14 = torch.ops.aten.view.default(input_158, input_126);  input_158 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(input_159, torch.bfloat16);  input_159 = None
        convert_element_type_default_19 = torch.ops.prims.convert_element_type.default(input_160, torch.bfloat16);  input_160 = None
        view_default_15 = torch.ops.aten.view.default(view_default_14, input_139);  view_default_14 = input_139 = None
        permute_default_6 = torch.ops.aten.permute.default(convert_element_type_default_19, [1, 0]);  convert_element_type_default_19 = None
        addmm_default_6 = torch.ops.aten.addmm.default(convert_element_type_default_18, view_default_15, permute_default_6, beta = 1, alpha = 1);  convert_element_type_default_18 = view_default_15 = permute_default_6 = None
        view_default_16 = torch.ops.aten.view.default(addmm_default_6, input_126);  addmm_default_6 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_10, view_default_16, alpha = 1);  view_default_16 = None
        convert_element_type_default_20 = torch.ops.prims.convert_element_type.default(add_tensor_13, torch.float32)
        var_mean_correction_6 = torch.ops.aten.var_mean.correction(convert_element_type_default_20, [2], correction = 0, keepdim = True)
        getitem_11 = var_mean_correction_6[0]
        getitem_12 = var_mean_correction_6[1];  var_mean_correction_6 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(getitem_11, 1e-05, alpha = 1);  getitem_11 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_14);  add_tensor_14 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(convert_element_type_default_20, getitem_12, alpha = 1);  convert_element_type_default_20 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_6, rsqrt_default_4);  sub_tensor_6 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, input_161);  mul_tensor_9 = input_161 = None
        convert_element_type_default_21 = torch.ops.prims.convert_element_type.default(input_162, torch.bfloat16);  input_162 = None
        view_default_17 = torch.ops.aten.view.default(input_163, input_133);  input_163 = input_133 = None
        permute_default_7 = torch.ops.aten.permute.default(input_164, [1, 0]);  input_164 = None
        addmm_default_7 = torch.ops.aten.addmm.default(convert_element_type_default_21, view_default_17, permute_default_7, beta = 1, alpha = 1);  convert_element_type_default_21 = view_default_17 = permute_default_7 = None
        view_default_18 = torch.ops.aten.view.default(addmm_default_7, input_126);  addmm_default_7 = input_126 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(add_tensor_13, view_default_18, alpha = 1);  view_default_18 = None
        convert_element_type_default_22 = torch.ops.prims.convert_element_type.default(add_tensor_15, torch.float32)
        var_mean_correction_7 = torch.ops.aten.var_mean.correction(convert_element_type_default_22, [2], correction = 0, keepdim = True)
        getitem_13 = var_mean_correction_7[0]
        getitem_14 = var_mean_correction_7[1];  var_mean_correction_7 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(convert_element_type_default_22, getitem_14, alpha = 1);  convert_element_type_default_22 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_7, input_165);  sub_tensor_7 = input_165 = None
        return (add_tensor, var_mean_correction, sub_tensor, add_tensor_1, getitem, getitem_1, mul_tensor, add_tensor_2, getitem_3, rsqrt_default, mul_tensor_2, add_tensor_4, getitem_5, rsqrt_default_1, add_tensor_6, add_tensor_7, getitem_8, rsqrt_default_2, convert_element_type_default_13, add_tensor_10, getitem_10, rsqrt_default_3, add_tensor_12, add_tensor_13, getitem_12, rsqrt_default_4, mul_tensor_10, add_tensor_15, getitem_13, getitem_14, mul_tensor_11)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
