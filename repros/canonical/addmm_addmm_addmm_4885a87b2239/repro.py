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
    def forward(self, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94):
        add_tensor = torch.ops.aten.add.Tensor(input_65, input_66, alpha = 1);  input_65 = input_66 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_67, input_68);  input_67 = input_68 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        view_default = torch.ops.aten.view.default(input_69, input_70);  input_69 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, input_71);  view_default = input_71 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        addmm_default = torch.ops.aten.addmm.default(input_72, input_73, input_74, beta = 1, alpha = 1);  input_72 = input_73 = input_74 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_70);  addmm_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_1, input_75);  view_default_1 = input_75 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_2, alpha = 1);  add_tensor_2 = mul_tensor_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor, input_76);  sub_tensor = input_76 = None
        view_default_2 = torch.ops.aten.view.default(input_77, input_78);  input_77 = None
        permute_default = torch.ops.aten.permute.default(input_79, [1, 0]);  input_79 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_80, view_default_2, permute_default, beta = 1, alpha = 1);  input_80 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_70);  addmm_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default_3, input_81);  view_default_3 = input_81 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_4, alpha = 1);  add_tensor_3 = mul_tensor_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_3, alpha = 1);  convert_element_type_default_1 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-06, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_82);  mul_tensor_5 = input_82 = None
        getitem_4 = input_83[0];  input_83 = None
        permute_default_1 = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_1, input_70);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_84);  view_default_4 = input_84 = None
        permute_default_2 = torch.ops.aten.permute.default(input_85, [1, 0]);  input_85 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_86, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_86 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_70);  addmm_default_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_6, input_87);  view_default_6 = input_87 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_7, alpha = 1);  add_tensor_4 = mul_tensor_7 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem_5 = var_mean_correction_2[0]
        getitem_6 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_6, alpha = 1);  convert_element_type_default_2 = getitem_6 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_5, 1e-06, alpha = 1);  getitem_5 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_1);  sub_tensor_2 = rsqrt_default_1 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, input_88);  mul_tensor_8 = input_88 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_9, input_89, alpha = 1);  mul_tensor_9 = input_89 = None
        view_default_7 = torch.ops.aten.view.default(input_90, input_78);  input_90 = input_78 = None
        permute_default_3 = torch.ops.aten.permute.default(input_91, [1, 0]);  input_91 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_92, view_default_7, permute_default_3, beta = 1, alpha = 1);  input_92 = view_default_7 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_70);  addmm_default_3 = input_70 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_default_8, input_93);  view_default_8 = input_93 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_6, mul_tensor_10, alpha = 1);  add_tensor_6 = mul_tensor_10 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_9, torch.float32);  add_tensor_9 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem_7 = var_mean_correction_3[0]
        getitem_8 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_8, alpha = 1);  convert_element_type_default_3 = getitem_8 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(getitem_7, 1e-06, alpha = 1);  getitem_7 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_2);  sub_tensor_3 = rsqrt_default_2 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, input_94);  mul_tensor_11 = input_94 = None
        return (getitem, mul_tensor_3, mul_tensor_6, add_tensor_8, mul_tensor_12)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
