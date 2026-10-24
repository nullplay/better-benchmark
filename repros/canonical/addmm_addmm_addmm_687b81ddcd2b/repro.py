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
    def forward(self, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154):
        add_tensor = torch.ops.aten.add.Tensor(input_108, input_109, alpha = 1);  input_108 = input_109 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_110, input_111);  input_110 = input_111 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        view_default = torch.ops.aten.view.default(input_112, input_113);  input_112 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_114, view_default);  input_114 = view_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        addmm_default = torch.ops.aten.addmm.default(input_115, input_116, input_117, beta = 1, alpha = 1);  input_115 = input_116 = input_117 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_113)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_118, view_default_1);  input_118 = view_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_2, alpha = 1);  add_tensor_2 = mul_tensor_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_3, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem, 1e-06, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_119);  input_119 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_120, torch.bfloat16);  input_120 = None
        view_default_2 = torch.ops.aten.view.default(input_121, input_122);  input_121 = None
        permute_default = torch.ops.aten.permute.default(input_123, [1, 0]);  input_123 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default, view_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_113)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_124, view_default_3);  input_124 = view_default_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_5, alpha = 1);  add_tensor_3 = mul_tensor_5 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-06, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_5, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_125);  input_125 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_7, input_126, alpha = 1);  mul_tensor_7 = input_126 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_127, torch.bfloat16);  input_127 = None
        view_default_4 = torch.ops.aten.view.default(input_128, input_122);  input_128 = None
        permute_default_1 = torch.ops.aten.permute.default(input_129, [1, 0]);  input_129 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_4, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_130);  input_130 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_9);  mul_tensor_9 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_8, add_tensor_8);  mul_tensor_8 = add_tensor_8 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_131, torch.bfloat16);  input_131 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_132, torch.bfloat16);  input_132 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_3, input_133);  convert_element_type_default_3 = input_133 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_6, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_3, input_113)
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_134, view_default_7);  input_134 = view_default_7 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_11, alpha = 1);  add_tensor_5 = mul_tensor_11 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-06, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_9, getitem_5, alpha = 1);  getitem_5 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, input_135);  input_135 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(mul_tensor_13, input_136, alpha = 1);  mul_tensor_13 = input_136 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.bfloat16);  add_tensor_11 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_137, torch.bfloat16);  input_137 = None
        view_default_8 = torch.ops.aten.view.default(input_138, input_122);  input_138 = input_122 = None
        permute_default_3 = torch.ops.aten.permute.default(input_139, [1, 0]);  input_139 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_7, view_default_8, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_7 = view_default_8 = permute_default_3 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default_4, input_113)
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_140, view_default_9);  input_140 = view_default_9 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_9, mul_tensor_14, alpha = 1);  add_tensor_9 = mul_tensor_14 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_12, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(getitem_6, 1e-06, alpha = 1);  getitem_6 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_13);  add_tensor_13 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_12, getitem_7, alpha = 1);  getitem_7 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_15, input_141);  input_141 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(mul_tensor_16, input_142, alpha = 1);  mul_tensor_16 = input_142 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_143, input_144, input_145, beta = 1, alpha = 1);  input_143 = input_144 = input_145 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_5, input_113)
        mul_tensor_17 = torch.ops.aten.mul.Tensor(input_146, view_default_10);  input_146 = view_default_10 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(add_tensor_12, mul_tensor_17, alpha = 1);  add_tensor_12 = mul_tensor_17 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(add_tensor_15, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_correction_4[0]
        getitem_9 = var_mean_correction_4[1];  var_mean_correction_4 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(getitem_8, 1e-06, alpha = 1);  getitem_8 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_16);  add_tensor_16 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_15, getitem_9, alpha = 1);  getitem_9 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_4);  sub_tensor_4 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(mul_tensor_18, input_147);  input_147 = None
        view_default_11 = torch.ops.aten.view.default(input_148, input_113);  input_148 = input_113 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(input_149, view_default_11);  input_149 = view_default_11 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_15, mul_tensor_20, alpha = 1);  add_tensor_15 = mul_tensor_20 = None
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(add_tensor_17, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_correction_5[0]
        getitem_11 = var_mean_correction_5[1];  var_mean_correction_5 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(add_tensor_17, getitem_11, alpha = 1);  getitem_11 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(sub_tensor_5, input_150);  sub_tensor_5 = input_150 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(input_151, input_152);  input_151 = input_152 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(add_tensor_17, mul_tensor_22, alpha = 1);  add_tensor_17 = mul_tensor_22 = None
        var_mean_correction_6 = torch.ops.aten.var_mean.correction(add_tensor_18, [2], correction = 0, keepdim = True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(add_tensor_18, input_153, alpha = 1);  input_153 = None
        add_tensor_19 = torch.ops.aten.add.Tensor(add_tensor_18, input_154, alpha = 1);  add_tensor_18 = input_154 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_3, 768);  rsqrt_default_3 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(rsqrt_default_2, 768);  rsqrt_default_2 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(rsqrt_default_1, 768);  rsqrt_default_1 = None
        return (addmm_default, rsqrt_default, mul_tensor_3, mul_tensor_4, addmm_default_1, mul_tensor_6, add_tensor_7, addmm_default_2, view_default_6, addmm_default_3, mul_tensor_12, convert_element_type_default_6, addmm_default_4, mul_tensor_15, add_tensor_14, addmm_default_5, rsqrt_default_4, mul_tensor_18, mul_tensor_19, getitem_10, mul_tensor_21, var_mean_correction_6, sub_tensor_6, add_tensor_19, div_tensor, div_tensor_1, div_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
