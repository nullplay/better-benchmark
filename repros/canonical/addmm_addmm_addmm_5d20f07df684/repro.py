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
    def forward(self, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118):
        view_default = torch.ops.aten.view.default(input_85, input_86);  input_85 = None
        view_default_1 = torch.ops.aten.view.default(input_87, input_86);  input_87 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, view_default_1, alpha = 1);  view_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor, input_88);  add_tensor = None
        addmm_default = torch.ops.aten.addmm.default(input_89, input_90, input_91, beta = 1, alpha = 1);  input_89 = input_90 = input_91 = None
        relu_default = torch.ops.aten.relu.default(addmm_default);  addmm_default = None
        permute_default = torch.ops.aten.permute.default(input_92, [1, 0]);  input_92 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_93, relu_default, permute_default, beta = 1, alpha = 1);  input_93 = relu_default = permute_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_2, addmm_default_1, alpha = 1);  view_default_2 = addmm_default_1 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_1, input_86);  add_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_94);  mul_tensor = input_94 = None
        permute_default_1 = torch.ops.aten.permute.default(input_95, [0, 2, 1, 3]);  input_95 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_1, input_96);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_97);  view_default_4 = None
        permute_default_2 = torch.ops.aten.permute.default(input_98, [1, 0]);  input_98 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_99, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_99 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_86);  addmm_default_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_3, view_default_6, alpha = 1);  view_default_3 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_3, input_88);  add_tensor_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [1], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_3, alpha = 1);  convert_element_type_default_1 = getitem_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_100);  mul_tensor_2 = input_100 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_3, input_101, alpha = 1);  mul_tensor_3 = input_101 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        permute_default_3 = torch.ops.aten.permute.default(input_102, [1, 0]);  input_102 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_103, convert_element_type_default_2, permute_default_3, beta = 1, alpha = 1);  input_103 = convert_element_type_default_2 = permute_default_3 = None
        relu_default_1 = torch.ops.aten.relu.default(addmm_default_3);  addmm_default_3 = None
        permute_default_4 = torch.ops.aten.permute.default(input_104, [1, 0]);  input_104 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_105, relu_default_1, permute_default_4, beta = 1, alpha = 1);  input_105 = relu_default_1 = permute_default_4 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_7, addmm_default_4, alpha = 1);  view_default_7 = addmm_default_4 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_6, input_86);  add_tensor_6 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_5, alpha = 1);  convert_element_type_default_3 = getitem_5 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_106);  mul_tensor_4 = input_106 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_5, input_107, alpha = 1);  mul_tensor_5 = input_107 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.bfloat16);  add_tensor_8 = None
        permute_default_5 = torch.ops.aten.permute.default(input_108, [0, 2, 1, 3]);  input_108 = None
        view_default_9 = torch.ops.aten.view.default(permute_default_5, input_96);  permute_default_5 = input_96 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_97);  view_default_9 = input_97 = None
        permute_default_6 = torch.ops.aten.permute.default(input_109, [1, 0]);  input_109 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_110, view_default_10, permute_default_6, beta = 1, alpha = 1);  input_110 = view_default_10 = permute_default_6 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_5, input_86);  addmm_default_5 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(view_default_8, view_default_11, alpha = 1);  view_default_8 = view_default_11 = None
        view_default_12 = torch.ops.aten.view.default(add_tensor_9, input_88);  add_tensor_9 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_12, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [1], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, getitem_7, alpha = 1);  convert_element_type_default_5 = getitem_7 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(getitem_6, 1e-05, alpha = 1);  getitem_6 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = rsqrt_default_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_111);  mul_tensor_6 = input_111 = None
        addmm_default_6 = torch.ops.aten.addmm.default(input_112, input_113, input_114, beta = 1, alpha = 1);  input_112 = input_113 = input_114 = None
        relu_default_2 = torch.ops.aten.relu.default(addmm_default_6);  addmm_default_6 = None
        permute_default_7 = torch.ops.aten.permute.default(input_115, [1, 0]);  input_115 = None
        addmm_default_7 = torch.ops.aten.addmm.default(input_116, relu_default_2, permute_default_7, beta = 1, alpha = 1);  input_116 = relu_default_2 = permute_default_7 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(view_default_12, addmm_default_7, alpha = 1);  view_default_12 = addmm_default_7 = None
        view_default_13 = torch.ops.aten.view.default(add_tensor_11, input_86);  add_tensor_11 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_13, torch.float32)
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convert_element_type_default_6, [2], correction = 0, keepdim = True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_6, input_117, alpha = 1);  convert_element_type_default_6 = input_117 = None
        view_default_14 = torch.ops.aten.view.default(input_118, input_86);  input_118 = input_86 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(view_default_13, view_default_14, alpha = 1);  view_default_13 = view_default_14 = None
        view_default_15 = torch.ops.aten.view.default(add_tensor_12, input_88);  add_tensor_12 = input_88 = None
        return (mul_tensor_1, convert_element_type_default_4, mul_tensor_7, var_mean_correction_4, sub_tensor_4, view_default_15)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
