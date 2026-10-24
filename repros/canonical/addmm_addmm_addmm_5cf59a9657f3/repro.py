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
    def forward(self, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128):
        add_tensor = torch.ops.aten.add.Tensor(input_98, input_99, alpha = 1);  input_98 = input_99 = None
        view_default = torch.ops.aten.view.default(input_100, input_101);  input_100 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default, add_tensor, alpha = 1);  view_default = add_tensor = None
        addmm_default = torch.ops.aten.addmm.default(input_102, input_103, input_104, beta = 1, alpha = 1);  input_102 = input_103 = input_104 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_101);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, input_105);  sub_tensor = input_105 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_106, torch.bfloat16);  input_106 = None
        view_default_2 = torch.ops.aten.view.default(input_107, input_108);  input_107 = None
        permute_default = torch.ops.aten.permute.default(input_109, [1, 0]);  input_109 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default, view_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_101);  addmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_3, add_tensor_2, alpha = 1);  view_default_3 = add_tensor_2 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_3, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_110);  input_110 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_111, torch.bfloat16);  input_111 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_112, torch.bfloat16);  input_112 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_113, torch.bfloat16);  input_113 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_3, input_114);  convert_element_type_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_4, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_101);  addmm_default_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_5, alpha = 1);  add_tensor_3 = view_default_5 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_5, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_5, getitem_5, alpha = 1);  getitem_5 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_1);  sub_tensor_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_115);  input_115 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_4, input_116, alpha = 1);  mul_tensor_4 = input_116 = None
        clone_default = torch.ops.aten.clone.default(input_117, memory_format = torch.contiguous_format);  input_117 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_101);  clone_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_118, torch.bfloat16);  input_118 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_119, torch.bfloat16);  input_119 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_108);  view_default_6 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_7, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_4 = view_default_7 = permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_101);  addmm_default_3 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_8, add_tensor_5, alpha = 1);  view_default_8 = add_tensor_5 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_8, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem_6, 1e-05, alpha = 1);  getitem_6 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_8, getitem_7, alpha = 1);  getitem_7 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_2);  sub_tensor_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_120);  input_120 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor_6, input_121, alpha = 1);  mul_tensor_6 = input_121 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_122, torch.bfloat16);  input_122 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_123, torch.bfloat16);  input_123 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.bfloat16);  add_tensor_10 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_8, input_108);  convert_element_type_default_8 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_9, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_3 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_4, input_124);  input_124 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_10, 0.5)
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_9, 3.0);  convert_element_type_default_9 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_11 = torch.ops.aten.add.Tensor(view_default_10, mul_tensor_8, alpha = 1);  view_default_10 = mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor_11, 0.7978845608028654);  add_tensor_11 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_9);  mul_tensor_9 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_7, add_tensor_12);  mul_tensor_7 = add_tensor_12 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_125, torch.bfloat16);  input_125 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(input_126, torch.bfloat16);  input_126 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        view_default_11 = torch.ops.aten.view.default(convert_element_type_default_12, input_114);  convert_element_type_default_12 = input_114 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_11, [1, 0]);  convert_element_type_default_11 = None
        addmm_default_5 = torch.ops.aten.addmm.default(convert_element_type_default_10, view_default_11, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_10 = permute_default_4 = None
        view_default_12 = torch.ops.aten.view.default(addmm_default_5, input_101);  addmm_default_5 = input_101 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_8, view_default_12, alpha = 1);  add_tensor_8 = view_default_12 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(add_tensor_13, [2], correction = 0, keepdim = True)
        getitem_8 = var_mean_correction_4[0]
        getitem_9 = var_mean_correction_4[1];  var_mean_correction_4 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(getitem_8, 1e-05, alpha = 1);  getitem_8 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_14);  add_tensor_14 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_13, getitem_9, alpha = 1);  add_tensor_13 = getitem_9 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_3);  sub_tensor_4 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, input_127);  input_127 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(mul_tensor_12, input_128, alpha = 1);  mul_tensor_12 = input_128 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(add_tensor_15, torch.bfloat16);  add_tensor_15 = None
        view_default_13 = torch.ops.aten.view.default(convert_element_type_default_13, input_108);  convert_element_type_default_13 = input_108 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_3, 2048);  rsqrt_default_3 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(rsqrt_default_2, 2048);  rsqrt_default_2 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(rsqrt_default_1, 2048);  rsqrt_default_1 = None
        return (getitem, mul_tensor, rsqrt_default, mul_tensor_1, mul_tensor_2, mul_tensor_3, add_tensor_7, mul_tensor_5, view_default_9, addmm_default_4, view_default_11, mul_tensor_11, view_default_13, div_tensor, div_tensor_1, div_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
