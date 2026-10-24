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
    def forward(self, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147):
        add_tensor = torch.ops.aten.add.Tensor(input_106, input_107, alpha = 1);  input_106 = input_107 = None
        view_default = torch.ops.aten.view.default(input_108, input_109);  input_108 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default, add_tensor, alpha = 1);  view_default = add_tensor = None
        addmm_default = torch.ops.aten.addmm.default(input_110, input_111, input_112, beta = 1, alpha = 1);  input_110 = input_111 = input_112 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_109);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_113, alpha = 1);  convert_element_type_default = input_113 = None
        view_default_2 = torch.ops.aten.view.default(input_114, input_115);  input_114 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_116, view_default_2, input_117, beta = 1, alpha = 1);  input_116 = view_default_2 = input_117 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_109);  addmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_3, add_tensor_2, alpha = 1);  view_default_3 = add_tensor_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction_1[0]
        getitem_1 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_1, alpha = 1);  convert_element_type_default_1 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor_1, input_118);  sub_tensor_1 = input_118 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_119, input_120);  input_119 = input_120 = None
        view_default_4 = torch.ops.aten.view.default(mul_tensor_1, input_121);  mul_tensor_1 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_122, view_default_4, input_123, beta = 1, alpha = 1);  input_122 = view_default_4 = input_123 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_109);  addmm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_5, alpha = 1);  add_tensor_3 = view_default_5 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_2[0]
        getitem_3 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_3, alpha = 1);  convert_element_type_default_2 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_124);  mul_tensor_2 = input_124 = None
        permute_default = torch.ops.aten.permute.default(input_125, [0, 2, 1, 3]);  input_125 = None
        view_default_6 = torch.ops.aten.view.default(permute_default, input_126);  permute_default = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_115);  view_default_6 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_127, view_default_7, input_128, beta = 1, alpha = 1);  input_127 = view_default_7 = input_128 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_109);  addmm_default_3 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_8, add_tensor_4, alpha = 1);  view_default_8 = add_tensor_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_3[0]
        getitem_5 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_5, alpha = 1);  convert_element_type_default_3 = getitem_5 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_1);  sub_tensor_3 = rsqrt_default_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_129);  mul_tensor_4 = input_129 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_5, input_130, alpha = 1);  mul_tensor_5 = input_130 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.bfloat16);  add_tensor_8 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_4, input_115);  convert_element_type_default_4 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_131, view_default_9, input_132, beta = 1, alpha = 1);  input_131 = view_default_9 = input_132 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_4, input_133);  addmm_default_4 = input_133 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(view_default_10, 0.5)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(view_default_10, 3.0)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_9 = torch.ops.aten.add.Tensor(view_default_10, mul_tensor_7, alpha = 1);  view_default_10 = mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_tensor_9, 0.7978845608028654);  add_tensor_9 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_8);  mul_tensor_8 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_6, add_tensor_10);  mul_tensor_6 = add_tensor_10 = None
        view_default_11 = torch.ops.aten.view.default(mul_tensor_9, input_121);  mul_tensor_9 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_134, view_default_11, input_135, beta = 1, alpha = 1);  input_134 = view_default_11 = input_135 = None
        view_default_12 = torch.ops.aten.view.default(addmm_default_5, input_109);  addmm_default_5 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_6, view_default_12, alpha = 1);  add_tensor_6 = view_default_12 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.float32)
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_4[0]
        getitem_7 = var_mean_correction_4[1];  var_mean_correction_4 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, getitem_7, alpha = 1);  convert_element_type_default_5 = getitem_7 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(getitem_6, 1e-05, alpha = 1);  getitem_6 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_2);  sub_tensor_4 = rsqrt_default_2 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, input_136);  mul_tensor_10 = input_136 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(mul_tensor_11, input_137, alpha = 1);  mul_tensor_11 = input_137 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_13, torch.bfloat16);  add_tensor_13 = None
        permute_default_1 = torch.ops.aten.permute.default(input_138, [0, 2, 1, 3]);  input_138 = None
        view_default_13 = torch.ops.aten.view.default(permute_default_1, input_126);  permute_default_1 = input_126 = None
        view_default_14 = torch.ops.aten.view.default(view_default_13, input_115);  view_default_13 = input_115 = None
        addmm_default_6 = torch.ops.aten.addmm.default(input_139, view_default_14, input_140, beta = 1, alpha = 1);  input_139 = view_default_14 = input_140 = None
        view_default_15 = torch.ops.aten.view.default(addmm_default_6, input_109);  addmm_default_6 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(view_default_15, add_tensor_11, alpha = 1);  view_default_15 = add_tensor_11 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_14, torch.float32)
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convert_element_type_default_7, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_correction_5[0]
        getitem_9 = var_mean_correction_5[1];  var_mean_correction_5 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_7, getitem_9, alpha = 1);  convert_element_type_default_7 = getitem_9 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(getitem_8, 1e-05, alpha = 1);  getitem_8 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_15);  add_tensor_15 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_default_3);  sub_tensor_5 = rsqrt_default_3 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, input_141);  mul_tensor_12 = input_141 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(mul_tensor_13, input_142, alpha = 1);  mul_tensor_13 = input_142 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_143, input_144);  input_143 = input_144 = None
        view_default_16 = torch.ops.aten.view.default(mul_tensor_14, input_121);  mul_tensor_14 = input_121 = None
        addmm_default_7 = torch.ops.aten.addmm.default(input_145, view_default_16, input_146, beta = 1, alpha = 1);  input_145 = view_default_16 = input_146 = None
        view_default_17 = torch.ops.aten.view.default(addmm_default_7, input_109);  addmm_default_7 = input_109 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_14, view_default_17, alpha = 1);  add_tensor_14 = view_default_17 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_17, torch.float32);  add_tensor_17 = None
        var_mean_correction_6 = torch.ops.aten.var_mean.correction(convert_element_type_default_8, [2], correction = 0, keepdim = True)
        getitem_10 = var_mean_correction_6[0]
        getitem_11 = var_mean_correction_6[1];  var_mean_correction_6 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(convert_element_type_default_8, getitem_11, alpha = 1);  convert_element_type_default_8 = getitem_11 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(getitem_10, 1e-05, alpha = 1);  getitem_10 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_18);  add_tensor_18 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_6, rsqrt_default_4);  sub_tensor_6 = rsqrt_default_4 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_15, input_147);  mul_tensor_15 = input_147 = None
        return (var_mean_correction, sub_tensor, getitem, mul_tensor, mul_tensor_3, convert_element_type_default_6, add_tensor_16, mul_tensor_16)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
