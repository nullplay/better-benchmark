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
    def forward(self, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123):
        add_tensor = torch.ops.aten.add.Tensor(input_89, input_90, alpha = 1);  input_89 = input_90 = None
        view_default = torch.ops.aten.view.default(input_91, input_92);  input_91 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        addmm_default = torch.ops.aten.addmm.default(input_93, input_94, input_95, beta = 1, alpha = 1);  input_93 = input_94 = input_95 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_92);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_96, alpha = 1);  convert_element_type_default = input_96 = None
        view_default_2 = torch.ops.aten.view.default(input_97, input_98);  input_97 = None
        permute_default = torch.ops.aten.permute.default(input_99, [1, 0]);  input_99 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_100, view_default_2, permute_default, beta = 1, alpha = 1);  input_100 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_92);  addmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction_1[0]
        getitem_1 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_1, alpha = 1);  convert_element_type_default_1 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor_1, input_101);  sub_tensor_1 = input_101 = None
        view_default_4 = torch.ops.aten.view.default(input_102, input_92);  input_102 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_103);  view_default_4 = None
        permute_default_1 = torch.ops.aten.permute.default(input_104, [1, 0]);  input_104 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_105, view_default_5, permute_default_1, beta = 1, alpha = 1);  input_105 = view_default_5 = permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_92);  addmm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_6, alpha = 1);  add_tensor_3 = view_default_6 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_2[0]
        getitem_3 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_3, alpha = 1);  convert_element_type_default_2 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-12, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_106);  mul_tensor_1 = input_106 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_107, input_108);  input_107 = input_108 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_3, input_98);  convert_element_type_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(input_109, [1, 0]);  input_109 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_110, view_default_7, permute_default_2, beta = 1, alpha = 1);  input_110 = view_default_7 = permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_92);  addmm_default_3 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_4, view_default_8, alpha = 1);  add_tensor_4 = view_default_8 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_3[0]
        getitem_5 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_5, alpha = 1);  convert_element_type_default_4 = getitem_5 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_4, 1e-12, alpha = 1);  getitem_4 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_1);  sub_tensor_3 = rsqrt_default_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_111);  mul_tensor_4 = input_111 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_5, input_112, alpha = 1);  mul_tensor_5 = input_112 = None
        getitem_6 = input_113[0];  input_113 = None
        permute_default_3 = torch.ops.aten.permute.default(getitem_6, [0, 2, 1, 3]);  getitem_6 = None
        clone_default = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_9 = torch.ops.aten.view.default(clone_default, input_92);  clone_default = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_103);  view_default_9 = input_103 = None
        permute_default_4 = torch.ops.aten.permute.default(input_114, [1, 0]);  input_114 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_115, view_default_10, permute_default_4, beta = 1, alpha = 1);  input_115 = view_default_10 = permute_default_4 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_4, input_92);  addmm_default_4 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_6, view_default_11, alpha = 1);  add_tensor_6 = view_default_11 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_9, torch.float32)
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [2], correction = 0, keepdim = True)
        getitem_7 = var_mean_correction_4[0]
        getitem_8 = var_mean_correction_4[1];  var_mean_correction_4 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, getitem_8, alpha = 1);  convert_element_type_default_5 = getitem_8 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(getitem_7, 1e-12, alpha = 1);  getitem_7 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_2);  sub_tensor_4 = rsqrt_default_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_116);  mul_tensor_6 = input_116 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(mul_tensor_7, input_117, alpha = 1);  mul_tensor_7 = input_117 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.bfloat16);  add_tensor_11 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_118, input_119);  input_118 = input_119 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_7, input_98);  convert_element_type_default_7 = input_98 = None
        permute_default_5 = torch.ops.aten.permute.default(input_120, [1, 0]);  input_120 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_121, view_default_12, permute_default_5, beta = 1, alpha = 1);  input_121 = view_default_12 = permute_default_5 = None
        view_default_13 = torch.ops.aten.view.default(addmm_default_5, input_92);  addmm_default_5 = input_92 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_9, view_default_13, alpha = 1);  add_tensor_9 = view_default_13 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_12, torch.float32);  add_tensor_12 = None
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convert_element_type_default_8, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_correction_5[0]
        getitem_10 = var_mean_correction_5[1];  var_mean_correction_5 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_8, getitem_10, alpha = 1);  convert_element_type_default_8 = getitem_10 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(getitem_9, 1e-12, alpha = 1);  getitem_9 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_13);  add_tensor_13 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_5, rsqrt_default_3);  sub_tensor_5 = rsqrt_default_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, input_122);  mul_tensor_9 = input_122 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(mul_tensor_10, input_123, alpha = 1);  mul_tensor_10 = input_123 = None
        return (var_mean_correction, sub_tensor, getitem, mul_tensor, mul_tensor_2, add_tensor_8, convert_element_type_default_6, add_tensor_14)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
