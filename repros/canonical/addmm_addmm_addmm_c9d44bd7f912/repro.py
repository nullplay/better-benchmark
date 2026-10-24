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
    def forward(self, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_71, torch.bfloat16);  input_71 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_72, torch.bfloat16);  input_72 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_73, torch.bfloat16);  input_73 = None
        convolution_default = torch.ops.aten.convolution.default(convert_element_type_default_2, convert_element_type_default_1, convert_element_type_default, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_default_2 = convert_element_type_default_1 = convert_element_type_default = None
        view_default = torch.ops.aten.view.default(convolution_default, input_74);  convolution_default = input_74 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1]);  view_default = None
        expand_default = torch.ops.aten.expand.default(input_75, input_76, implicit = False);  input_75 = input_76 = None
        cat_default = torch.ops.aten.cat.default([expand_default, permute_default], 1);  expand_default = permute_default = None
        getitem = input_77[0];  input_77 = None
        permute_default_1 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default_1 = torch.ops.aten.view.default(permute_default_1, input_78);  permute_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_79, torch.bfloat16);  input_79 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_80, torch.bfloat16);  input_80 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_81);  view_default_1 = input_81 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_2, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_3 = view_default_2 = permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default, input_78)
        mul_tensor = torch.ops.aten.mul.Tensor(input_82, view_default_3);  input_82 = view_default_3 = None
        add_tensor = torch.ops.aten.add.Tensor(cat_default, mul_tensor, alpha = 1);  cat_default = mul_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem_1 = var_mean_correction[0]
        getitem_2 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem_1, 1e-06, alpha = 1);  getitem_1 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, getitem_2, alpha = 1)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_83);  mul_tensor_1 = input_83 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_2, input_84, alpha = 1);  mul_tensor_2 = input_84 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_85, torch.bfloat16);  input_85 = None
        view_default_4 = torch.ops.aten.view.default(input_86, input_87);  input_86 = input_87 = None
        permute_default_3 = torch.ops.aten.permute.default(input_88, [1, 0]);  input_88 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_4, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_6 = view_default_4 = permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_1, input_78)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_89, view_default_5);  input_89 = view_default_5 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_3, alpha = 1);  add_tensor = mul_tensor_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_3, [2], correction = 0, keepdim = True)
        getitem_3 = var_mean_correction_1[0]
        getitem_4 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem_3, 1e-06, alpha = 1);  getitem_3 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_4, alpha = 1);  getitem_4 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_90);  input_90 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_5, input_91, alpha = 1);  mul_tensor_5 = input_91 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_92, input_93, input_94, beta = 1, alpha = 1);  input_92 = input_93 = input_94 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_78)
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_95, view_default_6);  input_95 = view_default_6 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6, alpha = 1);  add_tensor_3 = mul_tensor_6 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_6, [2], correction = 0, keepdim = True)
        getitem_5 = var_mean_correction_2[0]
        getitem_6 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_5, 1e-06, alpha = 1);  getitem_5 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_6, getitem_6, alpha = 1);  getitem_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_96);  input_96 = None
        view_default_7 = torch.ops.aten.view.default(input_97, input_78);  input_97 = input_78 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(input_98, view_default_7);  input_98 = view_default_7 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_6, mul_tensor_9, alpha = 1);  add_tensor_6 = mul_tensor_9 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_8, [2], correction = 0, keepdim = True)
        getitem_7 = var_mean_correction_3[0]
        getitem_8 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_8, getitem_8, alpha = 1);  getitem_8 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_99);  sub_tensor_3 = input_99 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_100, input_101);  input_100 = input_101 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_8, mul_tensor_11, alpha = 1);  add_tensor_8 = mul_tensor_11 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(add_tensor_9, [2], correction = 0, keepdim = True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_9, input_102, alpha = 1);  input_102 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_9, input_103, alpha = 1);  add_tensor_9 = input_103 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_1, 768);  rsqrt_default_1 = None
        return (addmm_default, getitem_2, rsqrt_default, convert_element_type_default_5, addmm_default_1, mul_tensor_4, add_tensor_5, addmm_default_2, rsqrt_default_2, mul_tensor_7, mul_tensor_8, getitem_7, mul_tensor_10, var_mean_correction_4, sub_tensor_4, add_tensor_10, div_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
