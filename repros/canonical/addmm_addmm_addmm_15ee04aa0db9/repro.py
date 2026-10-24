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
    def forward(self, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105):
        add_tensor = torch.ops.aten.add.Tensor(input_74, input_75, alpha = 1);  input_74 = input_75 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_76, input_77);  input_76 = input_77 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        view_default = torch.ops.aten.view.default(input_78, input_79);  input_78 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, input_80);  view_default = input_80 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        addmm_default = torch.ops.aten.addmm.default(input_81, input_82, input_83, beta = 1, alpha = 1);  input_81 = input_82 = input_83 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_79)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_1, input_84);  view_default_1 = input_84 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_2, alpha = 1);  add_tensor_2 = mul_tensor_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_3, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem, 1e-06, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_85);  input_85 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_86, torch.bfloat16);  input_86 = None
        view_default_2 = torch.ops.aten.view.default(input_87, input_88);  input_87 = None
        permute_default = torch.ops.aten.permute.default(input_89, [1, 0]);  input_89 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default, view_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_79)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_3, input_90);  view_default_3 = input_90 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_5, alpha = 1);  add_tensor_3 = mul_tensor_5 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_5, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-06, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_5, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_91);  input_91 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_7, input_92, alpha = 1);  mul_tensor_7 = input_92 = None
        getitem_4 = input_93[0];  input_93 = None
        permute_default_1 = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_1, input_79);  permute_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_94, torch.bfloat16);  input_94 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_95, torch.bfloat16);  input_95 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_96);  view_default_4 = input_96 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_5, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_79)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_default_6, input_97);  view_default_6 = input_97 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_5, mul_tensor_8, alpha = 1);  add_tensor_5 = mul_tensor_8 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_8, [2], correction = 0, keepdim = True)
        getitem_5 = var_mean_correction_2[0]
        getitem_6 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem_5, 1e-06, alpha = 1);  getitem_5 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_8, getitem_6, alpha = 1);  getitem_6 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, input_98);  input_98 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor_10, input_99, alpha = 1);  mul_tensor_10 = input_99 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.bfloat16);  add_tensor_10 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_100, torch.bfloat16);  input_100 = None
        view_default_7 = torch.ops.aten.view.default(input_101, input_88);  input_101 = input_88 = None
        permute_default_3 = torch.ops.aten.permute.default(input_102, [1, 0]);  input_102 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_7, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_4 = view_default_7 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_79);  input_79 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(view_default_8, input_103);  view_default_8 = input_103 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_8, mul_tensor_11, alpha = 1);  add_tensor_8 = mul_tensor_11 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_11, [2], correction = 0, keepdim = True)
        getitem_7 = var_mean_correction_3[0]
        getitem_8 = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(getitem_7, 1e-06, alpha = 1);  getitem_7 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_11, getitem_8, alpha = 1);  add_tensor_11 = getitem_8 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, input_104);  input_104 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(mul_tensor_13, input_105, alpha = 1);  mul_tensor_13 = input_105 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_3, 768);  rsqrt_default_3 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(rsqrt_default_2, 768);  rsqrt_default_2 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(rsqrt_default_1, 768);  rsqrt_default_1 = None
        return (addmm_default, rsqrt_default, mul_tensor_3, mul_tensor_4, addmm_default_1, mul_tensor_6, add_tensor_7, addmm_default_2, mul_tensor_9, convert_element_type_default_3, addmm_default_3, mul_tensor_12, add_tensor_13, div_tensor, div_tensor_1, div_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
