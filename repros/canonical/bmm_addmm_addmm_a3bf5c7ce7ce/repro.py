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
    def forward(self, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135):
        embedding_default = torch.ops.aten.embedding.default(input_93, input_94, 0, False, False);  input_93 = input_94 = None
        slice_tensor = torch.ops.aten.slice.Tensor(input_95, 1, 0, 128, 1);  input_95 = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, slice_tensor, alpha = 1);  embedding_default = slice_tensor = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_96, input_97, 0, False, False);  input_96 = input_97 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, embedding_default_1, alpha = 1);  add_tensor = embedding_default_1 = None
        expand_default = torch.ops.aten.expand.default(input_98, input_99, implicit = False);  input_98 = input_99 = None
        view_default = torch.ops.aten.view.default(expand_default, input_100);  expand_default = input_100 = None
        clone_default = torch.ops.aten.clone.default(input_101, memory_format = torch.contiguous_format);  input_101 = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_102);  clone_default = input_102 = None
        bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default, input_103);  bmm_default = input_103 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default_1, input_104);  clone_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_105);  view_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(input_106, [1, 0]);  input_106 = None
        addmm_default = torch.ops.aten.addmm.default(input_107, view_default_4, permute_default_1, beta = 1, alpha = 1);  input_107 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default, input_108);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_5, alpha = 1);  add_tensor_1 = view_default_5 = None
        mean_dim = torch.ops.aten.mean.dim(add_tensor_2, [-1], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, mean_dim, alpha = 1);  mean_dim = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_109, sub_tensor);  input_109 = sub_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_correction = torch.ops.aten.var.correction(convert_element_type_default, [-1], correction = 1.0, keepdim = True);  convert_element_type_default = None
        sqrt_default = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(sqrt_default, torch.bfloat16);  sqrt_default = None
        add_tensor_3 = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-06, alpha = 1);  convert_element_type_default_1 = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_3);  mul_tensor = add_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(div_tensor, input_110, alpha = 1);  div_tensor = input_110 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_4, input_105);  add_tensor_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_111, view_default_6, input_112, beta = 1, alpha = 1);  input_111 = view_default_6 = input_112 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_113, input_114);  input_113 = input_114 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_2, input_115);  convert_element_type_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(input_116, [1, 0]);  input_116 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_117, view_default_7, permute_default_2, beta = 1, alpha = 1);  input_117 = view_default_7 = permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_2, input_108);  addmm_default_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_8, alpha = 1);  add_tensor_2 = view_default_8 = None
        mean_dim_1 = torch.ops.aten.mean.dim(add_tensor_5, [-1], True, dtype = None)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_5, mean_dim_1, alpha = 1);  mean_dim_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_118, sub_tensor_1);  input_118 = sub_tensor_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        var_correction_1 = torch.ops.aten.var.correction(convert_element_type_default_3, [-1], correction = 1.0, keepdim = True);  convert_element_type_default_3 = None
        sqrt_default_1 = torch.ops.aten.sqrt.default(var_correction_1);  var_correction_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(sqrt_default_1, torch.bfloat16);  sqrt_default_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(convert_element_type_default_4, 1e-06, alpha = 1);  convert_element_type_default_4 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(mul_tensor_2, add_tensor_6);  mul_tensor_2 = add_tensor_6 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(div_tensor_1, input_119, alpha = 1);  div_tensor_1 = input_119 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_7, input_105)
        view_default_10 = torch.ops.aten.view.default(add_tensor_7, input_105)
        view_default_11 = torch.ops.aten.view.default(add_tensor_7, input_105);  add_tensor_7 = None
        view_default_12 = torch.ops.aten.view.default(input_120, input_104);  input_120 = input_104 = None
        view_default_13 = torch.ops.aten.view.default(view_default_12, input_105);  view_default_12 = input_105 = None
        permute_default_3 = torch.ops.aten.permute.default(input_121, [1, 0]);  input_121 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_122, view_default_13, permute_default_3, beta = 1, alpha = 1);  input_122 = view_default_13 = permute_default_3 = None
        view_default_14 = torch.ops.aten.view.default(addmm_default_3, input_108);  addmm_default_3 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_14, alpha = 1);  add_tensor_5 = view_default_14 = None
        mean_dim_2 = torch.ops.aten.mean.dim(add_tensor_8, [-1], True, dtype = None)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_8, mean_dim_2, alpha = 1);  mean_dim_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_123, sub_tensor_2);  input_123 = sub_tensor_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.float32)
        var_correction_2 = torch.ops.aten.var.correction(convert_element_type_default_5, [-1], correction = 1.0, keepdim = True);  convert_element_type_default_5 = None
        sqrt_default_2 = torch.ops.aten.sqrt.default(var_correction_2);  var_correction_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(sqrt_default_2, torch.bfloat16);  sqrt_default_2 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(convert_element_type_default_6, 1e-06, alpha = 1);  convert_element_type_default_6 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(mul_tensor_3, add_tensor_9);  mul_tensor_3 = add_tensor_9 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(div_tensor_2, input_124, alpha = 1);  div_tensor_2 = input_124 = None
        view_default_15 = torch.ops.aten.view.default(input_125, input_115);  input_125 = input_115 = None
        permute_default_4 = torch.ops.aten.permute.default(input_126, [1, 0]);  input_126 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_127, view_default_15, permute_default_4, beta = 1, alpha = 1);  input_127 = view_default_15 = permute_default_4 = None
        view_default_16 = torch.ops.aten.view.default(addmm_default_4, input_108);  addmm_default_4 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_8, view_default_16, alpha = 1);  add_tensor_8 = view_default_16 = None
        mean_dim_3 = torch.ops.aten.mean.dim(add_tensor_11, [-1], True, dtype = None)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_11, mean_dim_3, alpha = 1);  mean_dim_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_128, sub_tensor_3);  input_128 = sub_tensor_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.float32)
        var_correction_3 = torch.ops.aten.var.correction(convert_element_type_default_7, [-1], correction = 1.0, keepdim = True);  convert_element_type_default_7 = None
        sqrt_default_3 = torch.ops.aten.sqrt.default(var_correction_3);  var_correction_3 = None
        div_tensor_3 = torch.ops.aten.div.Tensor(mul_tensor_4, input_129);  mul_tensor_4 = input_129 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_130, input_131, input_132, beta = 1, alpha = 1);  input_130 = input_131 = input_132 = None
        view_default_17 = torch.ops.aten.view.default(addmm_default_5, input_108);  addmm_default_5 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_11, view_default_17, alpha = 1);  add_tensor_11 = view_default_17 = None
        mean_dim_4 = torch.ops.aten.mean.dim(add_tensor_12, [-1], True, dtype = None)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_12, mean_dim_4, alpha = 1);  mean_dim_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_133, sub_tensor_4);  input_133 = sub_tensor_4 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_12, torch.float32)
        var_correction_4 = torch.ops.aten.var.correction(convert_element_type_default_8, [-1], correction = 1.0, keepdim = True);  convert_element_type_default_8 = None
        view_default_18 = torch.ops.aten.view.default(input_134, input_108);  input_134 = input_108 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_12, view_default_18, alpha = 1);  add_tensor_12 = view_default_18 = None
        mean_dim_5 = torch.ops.aten.mean.dim(add_tensor_13, [-1], True, dtype = None)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(add_tensor_13, mean_dim_5, alpha = 1);  mean_dim_5 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor_13, torch.float32)
        add_tensor_14 = torch.ops.aten.add.Tensor(add_tensor_13, input_135, alpha = 1);  add_tensor_13 = input_135 = None
        return (addmm_default_1, view_default_9, view_default_10, view_default_11, add_tensor_10, sqrt_default_3, div_tensor_3, mul_tensor_5, var_correction_4, sub_tensor_5, convert_element_type_default_9, add_tensor_14)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
