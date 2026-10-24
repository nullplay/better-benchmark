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
    def forward(self, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82):
        add_tensor = torch.ops.aten.add.Tensor(input_56, input_57, alpha = 1);  input_56 = input_57 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_58, input_59);  input_58 = input_59 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        view_default = torch.ops.aten.view.default(input_60, input_61);  input_60 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_62, view_default);  input_62 = view_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        addmm_default = torch.ops.aten.addmm.default(input_63, input_64, input_65, beta = 1, alpha = 1);  input_63 = input_64 = input_65 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_61);  addmm_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_66, view_default_1);  input_66 = view_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_2, alpha = 1);  add_tensor_2 = mul_tensor_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor, input_67);  sub_tensor = input_67 = None
        view_default_2 = torch.ops.aten.view.default(input_68, input_69);  input_68 = None
        permute_default = torch.ops.aten.permute.default(input_70, [1, 0]);  input_70 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_71, view_default_2, permute_default, beta = 1, alpha = 1);  input_71 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_61);  addmm_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_72, view_default_3);  input_72 = view_default_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_4, alpha = 1);  add_tensor_3 = mul_tensor_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_3, alpha = 1);  convert_element_type_default_1 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-06, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_73);  mul_tensor_5 = input_73 = None
        view_default_4 = torch.ops.aten.view.default(input_74, input_69);  input_74 = input_69 = None
        permute_default_1 = torch.ops.aten.permute.default(input_75, [1, 0]);  input_75 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_76, view_default_4, permute_default_1, beta = 1, alpha = 1);  input_76 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_77);  addmm_default_2 = input_77 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_8);  mul_tensor_8 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_7, add_tensor_6);  mul_tensor_7 = add_tensor_6 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_3, input_78);  convert_element_type_default_3 = input_78 = None
        permute_default_2 = torch.ops.aten.permute.default(input_79, [1, 0]);  input_79 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_80, view_default_6, permute_default_2, beta = 1, alpha = 1);  input_80 = view_default_6 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_3, input_61);  addmm_default_3 = input_61 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_81, view_default_7);  input_81 = view_default_7 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_10, alpha = 1);  add_tensor_4 = mul_tensor_10 = None
        slice_tensor = torch.ops.aten.slice.Tensor(add_tensor_7, 1, 1, 9223372036854775807, 1);  add_tensor_7 = None
        mean_dim = torch.ops.aten.mean.dim(slice_tensor, [1], False, dtype = None);  slice_tensor = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mean_dim, torch.float32);  mean_dim = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [1], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_5, alpha = 1);  convert_element_type_default_4 = getitem_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_82);  sub_tensor_2 = input_82 = None
        return (getitem, mul_tensor_3, mul_tensor_6, getitem_4, mul_tensor_11)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
