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
    def forward(self, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146):
        permute_default = torch.ops.aten.permute.default(input_106, [0, 2, 1]);  input_106 = None
        embedding_default = torch.ops.aten.embedding.default(input_107, input_108, -1, False, False);  input_107 = input_108 = None
        add_tensor = torch.ops.aten.add.Tensor(permute_default, embedding_default, alpha = 1);  permute_default = embedding_default = None
        addmm_default = torch.ops.aten.addmm.default(input_109, input_110, input_111, beta = 1, alpha = 1);  input_109 = input_110 = input_111 = None
        view_default = torch.ops.aten.view.default(addmm_default, input_112);  addmm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        clone_default = torch.ops.aten.clone.default(add_tensor_1, memory_format = torch.contiguous_format)
        convert_element_type_default = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        view_default_1 = torch.ops.aten.view.default(input_113, input_114);  input_113 = None
        permute_default_1 = torch.ops.aten.permute.default(input_115, [1, 0]);  input_115 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_116, view_default_1, permute_default_1, beta = 1, alpha = 1);  input_116 = view_default_1 = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default_1, input_112);  addmm_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2, alpha = 1);  add_tensor_1 = view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(add_tensor_2, memory_format = torch.contiguous_format)
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(clone_default_1, torch.float32);  clone_default_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, input_117, alpha = 1);  convert_element_type_default_1 = input_117 = None
        view_default_3 = torch.ops.aten.view.default(input_118, input_119);  input_118 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_120);  view_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(input_121, [1, 0]);  input_121 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_122, view_default_4, permute_default_2, beta = 1, alpha = 1);  input_122 = view_default_4 = permute_default_2 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_112);  addmm_default_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_5, alpha = 1);  add_tensor_2 = view_default_5 = None
        clone_default_2 = torch.ops.aten.clone.default(add_tensor_3, memory_format = torch.contiguous_format)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(clone_default_2, torch.float32);  clone_default_2 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction_1[0]
        getitem_1 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1, alpha = 1);  convert_element_type_default_2 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor_1, input_123);  sub_tensor_1 = input_123 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_124, input_125);  input_124 = input_125 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_3, input_114);  convert_element_type_default_3 = None
        permute_default_3 = torch.ops.aten.permute.default(input_126, [1, 0]);  input_126 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_127, view_default_6, permute_default_3, beta = 1, alpha = 1);  input_127 = view_default_6 = permute_default_3 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_3, input_112);  addmm_default_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_7, alpha = 1);  add_tensor_3 = view_default_7 = None
        clone_default_3 = torch.ops.aten.clone.default(add_tensor_4, memory_format = torch.contiguous_format)
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(clone_default_3, torch.float32);  clone_default_3 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_2[0]
        getitem_3 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_3, alpha = 1);  convert_element_type_default_4 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_128);  mul_tensor_2 = input_128 = None
        getitem_4 = input_129[0];  input_129 = None
        permute_default_4 = torch.ops.aten.permute.default(getitem_4, [0, 2, 1, 3]);  getitem_4 = None
        clone_default_4 = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_4, input_119);  clone_default_4 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_120);  view_default_8 = None
        permute_default_5 = torch.ops.aten.permute.default(input_130, [1, 0]);  input_130 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_131, view_default_9, permute_default_5, beta = 1, alpha = 1);  input_131 = view_default_9 = permute_default_5 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_4, input_112);  addmm_default_4 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_4, view_default_10, alpha = 1);  add_tensor_4 = view_default_10 = None
        clone_default_5 = torch.ops.aten.clone.default(add_tensor_6, memory_format = torch.contiguous_format)
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(clone_default_5, torch.float32);  clone_default_5 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [2], correction = 0, keepdim = True)
        getitem_5 = var_mean_correction_3[0]
        getitem_6 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, getitem_6, alpha = 1);  convert_element_type_default_5 = getitem_6 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_5, 1e-05, alpha = 1);  getitem_5 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_1);  sub_tensor_3 = rsqrt_default_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_132);  mul_tensor_4 = input_132 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_5, input_133, alpha = 1);  mul_tensor_5 = input_133 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_134, input_135);  input_134 = input_135 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        view_default_11 = torch.ops.aten.view.default(convert_element_type_default_6, input_114);  convert_element_type_default_6 = None
        permute_default_6 = torch.ops.aten.permute.default(input_136, [1, 0]);  input_136 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_137, view_default_11, permute_default_6, beta = 1, alpha = 1);  input_137 = view_default_11 = permute_default_6 = None
        view_default_12 = torch.ops.aten.view.default(addmm_default_5, input_112);  addmm_default_5 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_6, view_default_12, alpha = 1);  add_tensor_6 = view_default_12 = None
        clone_default_6 = torch.ops.aten.clone.default(add_tensor_9, memory_format = torch.contiguous_format)
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(clone_default_6, torch.float32);  clone_default_6 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convert_element_type_default_7, [2], correction = 0, keepdim = True)
        getitem_7 = var_mean_correction_4[0]
        getitem_8 = var_mean_correction_4[1];  var_mean_correction_4 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_7, getitem_8, alpha = 1);  convert_element_type_default_7 = getitem_8 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(getitem_7, 1e-05, alpha = 1);  getitem_7 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_4, rsqrt_default_2);  sub_tensor_4 = rsqrt_default_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_138);  mul_tensor_7 = input_138 = None
        view_default_13 = torch.ops.aten.view.default(input_139, input_119);  input_139 = input_119 = None
        view_default_14 = torch.ops.aten.view.default(view_default_13, input_120);  view_default_13 = input_120 = None
        permute_default_7 = torch.ops.aten.permute.default(input_140, [1, 0]);  input_140 = None
        addmm_default_6 = torch.ops.aten.addmm.default(input_141, view_default_14, permute_default_7, beta = 1, alpha = 1);  input_141 = view_default_14 = permute_default_7 = None
        view_default_15 = torch.ops.aten.view.default(addmm_default_6, input_112);  addmm_default_6 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_9, view_default_15, alpha = 1);  add_tensor_9 = view_default_15 = None
        clone_default_7 = torch.ops.aten.clone.default(add_tensor_11, memory_format = torch.contiguous_format)
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(clone_default_7, torch.float32);  clone_default_7 = None
        var_mean_correction_5 = torch.ops.aten.var_mean.correction(convert_element_type_default_8, [2], correction = 0, keepdim = True)
        getitem_9 = var_mean_correction_5[0]
        getitem_10 = var_mean_correction_5[1];  var_mean_correction_5 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_8, getitem_10, alpha = 1);  convert_element_type_default_8 = getitem_10 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_5, input_142);  sub_tensor_5 = input_142 = None
        view_default_16 = torch.ops.aten.view.default(input_143, input_114);  input_143 = input_114 = None
        permute_default_8 = torch.ops.aten.permute.default(input_144, [1, 0]);  input_144 = None
        addmm_default_7 = torch.ops.aten.addmm.default(input_145, view_default_16, permute_default_8, beta = 1, alpha = 1);  input_145 = view_default_16 = permute_default_8 = None
        view_default_17 = torch.ops.aten.view.default(addmm_default_7, input_112);  addmm_default_7 = input_112 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_11, view_default_17, alpha = 1);  add_tensor_11 = view_default_17 = None
        clone_default_8 = torch.ops.aten.clone.default(add_tensor_12, memory_format = torch.contiguous_format);  add_tensor_12 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(clone_default_8, torch.float32);  clone_default_8 = None
        var_mean_correction_6 = torch.ops.aten.var_mean.correction(convert_element_type_default_9, [2], correction = 0, keepdim = True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(convert_element_type_default_9, input_146, alpha = 1);  convert_element_type_default_9 = input_146 = None
        return (convert_element_type_default, var_mean_correction, sub_tensor, getitem, mul_tensor, mul_tensor_3, add_tensor_8, mul_tensor_8, getitem_9, mul_tensor_9, var_mean_correction_6, sub_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
