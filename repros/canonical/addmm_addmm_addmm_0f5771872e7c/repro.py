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
    def forward(self, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116):
        clone_default = torch.ops.aten.clone.default(input_91, memory_format = torch.contiguous_format);  input_91 = None
        view_default = torch.ops.aten.view.default(clone_default, input_92);  clone_default = input_92 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 3, 2, 1]);  view_default = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default_1, input_93);  clone_default_1 = None
        clone_default_2 = torch.ops.aten.clone.default(input_94, memory_format = torch.contiguous_format);  input_94 = None
        view_default_2 = torch.ops.aten.view.default(clone_default_2, input_93);  clone_default_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_95, torch.bfloat16);  input_95 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_96, torch.bfloat16);  input_96 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_97);  view_default_2 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default = view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default, input_93);  addmm_default = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_1, view_default_4, alpha = 1);  view_default_1 = view_default_4 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1, alpha = 1);  convert_element_type_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_98);  mul_tensor = input_98 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_99, alpha = 1);  mul_tensor_1 = input_99 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_100, torch.bfloat16);  input_100 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_101, torch.bfloat16);  input_101 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_5, input_97);  convert_element_type_default_5 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_5, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_102);  input_102 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        neg_default = torch.ops.aten.neg.default(convert_element_type_default_6)
        exp_default = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_3 = torch.ops.aten.add.Tensor(exp_default, 1, alpha = 1);  exp_default = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_6, add_tensor_3);  convert_element_type_default_6 = add_tensor_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_103, torch.bfloat16);  input_103 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_104, torch.bfloat16);  input_104 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_7, input_105);  convert_element_type_default_7 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_9, [1, 0]);  convert_element_type_default_9 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_8, view_default_7, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_8 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_2, input_93);  addmm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor, view_default_8, alpha = 1);  view_default_8 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_10, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_10, getitem_3, alpha = 1);  convert_element_type_default_10 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_106);  mul_tensor_2 = input_106 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_3, input_107, alpha = 1);  mul_tensor_3 = input_107 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16);  add_tensor_6 = None
        clone_default_3 = torch.ops.aten.clone.default(input_108, memory_format = torch.contiguous_format);  input_108 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_3, input_93);  clone_default_3 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(input_109, torch.bfloat16);  input_109 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(input_110, torch.bfloat16);  input_110 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_97);  view_default_9 = input_97 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_13, [1, 0]);  convert_element_type_default_13 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_12, view_default_10, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_12 = view_default_10 = permute_default_4 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_3, input_93);  addmm_default_3 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_4, view_default_11, alpha = 1);  view_default_11 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_14, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_14, getitem_5, alpha = 1);  convert_element_type_default_14 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_111);  mul_tensor_4 = input_111 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(mul_tensor_5, input_112, alpha = 1);  mul_tensor_5 = input_112 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(input_113, torch.bfloat16);  input_113 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(input_114, torch.bfloat16);  input_114 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(input_115, torch.bfloat16);  input_115 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_15, input_105);  convert_element_type_default_15 = input_105 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_17, [1, 0]);  convert_element_type_default_17 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_16, view_default_12, permute_default_5, beta = 1, alpha = 1);  convert_element_type_default_16 = view_default_12 = permute_default_5 = None
        view_default_13 = torch.ops.aten.view.default(addmm_default_4, input_93);  addmm_default_4 = input_93 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_7, view_default_13, alpha = 1);  view_default_13 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_18, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(getitem_6, 1e-05, alpha = 1);  getitem_6 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_11);  add_tensor_11 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_18, getitem_7, alpha = 1);  convert_element_type_default_18 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_116);  mul_tensor_6 = input_116 = None
        return (add_tensor, getitem_1, rsqrt_default, view_default_5, addmm_default_1, view_default_7, add_tensor_4, getitem_3, rsqrt_default_1, convert_element_type_default_11, add_tensor_7, getitem_5, rsqrt_default_2, add_tensor_9, add_tensor_10, getitem_7, rsqrt_default_3, mul_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
