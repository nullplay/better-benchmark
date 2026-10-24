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
    def forward(self, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76):
        view_default = torch.ops.aten.view.default(input_52, input_53);  input_52 = None
        addmm_default = torch.ops.aten.addmm.default(input_54, input_55, input_56, beta = 1, alpha = 1);  input_54 = input_55 = input_56 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, addmm_default, alpha = 1);  view_default = addmm_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_57);  add_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_58, alpha = 1);  convert_element_type_default = input_58 = None
        view_default_2 = torch.ops.aten.view.default(input_59, input_60);  input_59 = None
        permute_default = torch.ops.aten.permute.default(input_61, [1, 0]);  input_61 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_62, view_default_2, permute_default, beta = 1, alpha = 1);  input_62 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_57);  addmm_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, view_default_3, alpha = 1);  view_default_1 = view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_1, input_53);  add_tensor_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [1], correction = 0, keepdim = True)
        getitem = var_mean_correction_1[0]
        getitem_1 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_1, alpha = 1);  convert_element_type_default_1 = getitem_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_63);  mul_tensor = input_63 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_64, alpha = 1);  mul_tensor_1 = input_64 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        permute_default_1 = torch.ops.aten.permute.default(input_65, [1, 0]);  input_65 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_66, convert_element_type_default_2, permute_default_1, beta = 1, alpha = 1);  input_66 = convert_element_type_default_2 = permute_default_1 = None
        relu_default = torch.ops.aten.relu.default(addmm_default_2);  addmm_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(input_67, [1, 0]);  input_67 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_68, relu_default, permute_default_2, beta = 1, alpha = 1);  input_68 = relu_default = permute_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_4, addmm_default_3, alpha = 1);  view_default_4 = addmm_default_3 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_4, input_57);  add_tensor_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_2[0]
        getitem_3 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_3, alpha = 1);  convert_element_type_default_3 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_1);  sub_tensor_2 = rsqrt_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_69);  mul_tensor_2 = input_69 = None
        view_default_6 = torch.ops.aten.view.default(input_70, input_60);  input_70 = input_60 = None
        permute_default_3 = torch.ops.aten.permute.default(input_71, [1, 0]);  input_71 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_72, view_default_6, permute_default_3, beta = 1, alpha = 1);  input_72 = view_default_6 = permute_default_3 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_4, input_57);  addmm_default_4 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_5, view_default_7, alpha = 1);  view_default_5 = view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_6, input_53);  add_tensor_6 = input_53 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [1], correction = 0, keepdim = True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, input_73, alpha = 1);  convert_element_type_default_4 = input_73 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_74, input_75, input_76, beta = 1, alpha = 1);  input_74 = input_75 = input_76 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(view_default_8, addmm_default_5, alpha = 1);  view_default_8 = addmm_default_5 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_7, input_57);  add_tensor_7 = input_57 = None
        return (var_mean_correction, sub_tensor, mul_tensor_3, var_mean_correction_3, sub_tensor_3, view_default_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
