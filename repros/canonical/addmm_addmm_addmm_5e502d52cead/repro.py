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
    def forward(self, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81):
        add_tensor = torch.ops.aten.add.Tensor(input_59, input_60, alpha = 1);  input_59 = input_60 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_61);  mul_tensor = input_61 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_62, alpha = 1);  mul_tensor_1 = input_62 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_63);  convert_element_type_default_1 = input_63 = None
        permute_default = torch.ops.aten.permute.default(input_64, [1, 0]);  input_64 = None
        addmm_default = torch.ops.aten.addmm.default(input_65, view_default, permute_default, beta = 1, alpha = 1);  input_65 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_66);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_67)
        permute_default_1 = torch.ops.aten.permute.default(input_68, [1, 0]);  input_68 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_69, view_default_2, permute_default_1, beta = 1, alpha = 1);  input_69 = view_default_2 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_66);  addmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_70);  view_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        mul_scalar = torch.ops.aten.mul.Scalar(permute_default_2, 0.3535533905932738);  permute_default_2 = None
        expand_default = torch.ops.aten.expand.default(mul_scalar, input_71, implicit = False);  mul_scalar = None
        view_default_5 = torch.ops.aten.view.default(view_default_1, input_67)
        permute_default_3 = torch.ops.aten.permute.default(input_72, [1, 0]);  input_72 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_73, view_default_5, permute_default_3, beta = 1, alpha = 1);  input_73 = view_default_5 = permute_default_3 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_66);  addmm_default_2 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_70);  view_default_6 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 1, 3, 2]);  permute_default_4 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(permute_default_5, 0.3535533905932738);  permute_default_5 = None
        view_default_8 = torch.ops.aten.view.default(view_default_1, input_67)
        permute_default_6 = torch.ops.aten.permute.default(input_74, [1, 0]);  input_74 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_75, view_default_8, permute_default_6, beta = 1, alpha = 1);  input_75 = view_default_8 = permute_default_6 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default_3, input_66);  addmm_default_3 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_70);  view_default_9 = input_70 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_7, input_71, implicit = False);  permute_default_7 = input_71 = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        permute_default_8 = torch.ops.aten.permute.default(input_76, [0, 2, 1, 3]);  input_76 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_8, memory_format = torch.contiguous_format);  permute_default_8 = None
        view_default_11 = torch.ops.aten.view.default(clone_default_1, input_77);  clone_default_1 = input_77 = None
        view_default_12 = torch.ops.aten.view.default(view_default_11, input_67);  view_default_11 = input_67 = None
        permute_default_9 = torch.ops.aten.permute.default(input_78, [1, 0]);  input_78 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_79, view_default_12, permute_default_9, beta = 1, alpha = 1);  input_79 = view_default_12 = permute_default_9 = None
        view_default_13 = torch.ops.aten.view.default(addmm_default_4, input_66);  addmm_default_4 = input_66 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_1, view_default_13, alpha = 1);  view_default_1 = view_default_13 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_3, alpha = 1);  convert_element_type_default_2 = getitem_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-12, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_80);  mul_tensor_2 = input_80 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_3, input_81, alpha = 1);  mul_tensor_3 = input_81 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        return (expand_default, mul_scalar_1, clone_default, convert_element_type_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
