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
    def forward(self, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130):
        add_tensor = torch.ops.aten.add.Tensor(input_94, input_95, alpha = 1);  input_94 = input_95 = None
        view_default = torch.ops.aten.view.default(input_96, input_97);  input_96 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        addmm_default = torch.ops.aten.addmm.default(input_98, input_99, input_100, beta = 1, alpha = 1);  input_98 = input_99 = input_100 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_97);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        mean_dim = torch.ops.aten.mean.dim(add_tensor_2, [-1], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, mean_dim, alpha = 1);  mean_dim = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_101, sub_tensor);  input_101 = sub_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_correction = torch.ops.aten.var.correction(convert_element_type_default, [-1], correction = 1.0, keepdim = True);  convert_element_type_default = None
        view_default_2 = torch.ops.aten.view.default(input_102, input_103);  input_102 = None
        permute_default = torch.ops.aten.permute.default(input_104, [1, 0]);  input_104 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_105, view_default_2, permute_default, beta = 1, alpha = 1);  input_105 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_97);  addmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        mean_dim_1 = torch.ops.aten.mean.dim(add_tensor_3, [-1], True, dtype = None)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_3, mean_dim_1, alpha = 1);  mean_dim_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_106, sub_tensor_1);  input_106 = sub_tensor_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        var_correction_1 = torch.ops.aten.var.correction(convert_element_type_default_1, [-1], correction = 1.0, keepdim = True);  convert_element_type_default_1 = None
        sqrt_default = torch.ops.aten.sqrt.default(var_correction_1);  var_correction_1 = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor_1, input_107);  mul_tensor_1 = input_107 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_108, torch.bfloat16);  input_108 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_2, input_109);  convert_element_type_default_2 = None
        permute_default_1 = torch.ops.aten.permute.default(input_110, [1, 0]);  input_110 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_111, view_default_4, permute_default_1, beta = 1, alpha = 1);  input_111 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_97);  addmm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_5, alpha = 1);  add_tensor_3 = view_default_5 = None
        mean_dim_2 = torch.ops.aten.mean.dim(add_tensor_4, [-1], True, dtype = None)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_4, mean_dim_2, alpha = 1);  mean_dim_2 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_112, sub_tensor_2);  input_112 = sub_tensor_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        var_correction_2 = torch.ops.aten.var.correction(convert_element_type_default_3, [-1], correction = 1.0, keepdim = True);  convert_element_type_default_3 = None
        sqrt_default_1 = torch.ops.aten.sqrt.default(var_correction_2);  var_correction_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(sqrt_default_1, torch.bfloat16);  sqrt_default_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(convert_element_type_default_4, 1e-06, alpha = 1);  convert_element_type_default_4 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(mul_tensor_2, add_tensor_5);  mul_tensor_2 = add_tensor_5 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(div_tensor_1, input_113, alpha = 1);  div_tensor_1 = input_113 = None
        clone_default = torch.ops.aten.clone.default(input_114, memory_format = torch.contiguous_format);  input_114 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_115);  clone_default = input_115 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_103);  view_default_6 = None
        permute_default_2 = torch.ops.aten.permute.default(input_116, [1, 0]);  input_116 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_117, view_default_7, permute_default_2, beta = 1, alpha = 1);  input_117 = view_default_7 = permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_97);  addmm_default_3 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_4, view_default_8, alpha = 1);  add_tensor_4 = view_default_8 = None
        mean_dim_3 = torch.ops.aten.mean.dim(add_tensor_7, [-1], True, dtype = None)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_7, mean_dim_3, alpha = 1);  mean_dim_3 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_118, sub_tensor_3);  input_118 = sub_tensor_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.float32)
        var_correction_3 = torch.ops.aten.var.correction(convert_element_type_default_5, [-1], correction = 1.0, keepdim = True);  convert_element_type_default_5 = None
        sqrt_default_2 = torch.ops.aten.sqrt.default(var_correction_3);  var_correction_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(sqrt_default_2, torch.bfloat16);  sqrt_default_2 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(convert_element_type_default_6, 1e-06, alpha = 1);  convert_element_type_default_6 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(mul_tensor_3, add_tensor_8);  mul_tensor_3 = add_tensor_8 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(div_tensor_2, input_119, alpha = 1);  div_tensor_2 = input_119 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_9, input_103);  add_tensor_9 = None
        permute_default_3 = torch.ops.aten.permute.default(input_120, [1, 0]);  input_120 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_121, view_default_9, permute_default_3, beta = 1, alpha = 1);  input_121 = view_default_9 = permute_default_3 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_4, input_122);  addmm_default_4 = input_122 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 0.5)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 0.7071067811865476);  convert_element_type_default_7 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_10);  mul_tensor_4 = add_tensor_10 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        view_default_11 = torch.ops.aten.view.default(convert_element_type_default_8, input_109);  convert_element_type_default_8 = input_109 = None
        permute_default_4 = torch.ops.aten.permute.default(input_123, [1, 0]);  input_123 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_124, view_default_11, permute_default_4, beta = 1, alpha = 1);  input_124 = view_default_11 = permute_default_4 = None
        view_default_12 = torch.ops.aten.view.default(addmm_default_5, input_97);  addmm_default_5 = input_97 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_7, view_default_12, alpha = 1);  add_tensor_7 = view_default_12 = None
        select_int = torch.ops.aten.select.int(add_tensor_11, 1, 0)
        permute_default_5 = torch.ops.aten.permute.default(input_125, [1, 0]);  input_125 = None
        addmm_default_6 = torch.ops.aten.addmm.default(input_126, select_int, permute_default_5, beta = 1, alpha = 1);  input_126 = select_int = permute_default_5 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(addmm_default_6, torch.float32);  addmm_default_6 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_9, [-1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_9, amax_default, alpha = 1);  convert_element_type_default_9 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_4)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(sub_tensor_4, log_default, alpha = 1);  sub_tensor_4 = log_default = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(sub_tensor_5, torch.bfloat16);  sub_tensor_5 = None
        full_default = torch.ops.aten.full.default(input_127, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_127 = None
        cat_default = torch.ops.aten.cat.default([input_128, full_default], 0);  input_128 = full_default = None
        view_default_13 = torch.ops.aten.view.default(add_tensor_11, input_103);  add_tensor_11 = input_103 = None
        permute_default_6 = torch.ops.aten.permute.default(input_129, [1, 0]);  input_129 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default_6, [0, 3, 0, 0], 0);  permute_default_6 = None
        addmm_default_7 = torch.ops.aten.addmm.default(cat_default, view_default_13, constant_pad_nd_default, beta = 1, alpha = 1);  cat_default = view_default_13 = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(addmm_default_7, 1, 0, -3, 1);  addmm_default_7 = None
        view_default_14 = torch.ops.aten.view.default(slice_tensor, input_130);  slice_tensor = input_130 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(view_default_14, torch.float32);  view_default_14 = None
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default_11, [-1], True)
        sub_tensor_6 = torch.ops.aten.sub.Tensor(convert_element_type_default_11, amax_default_1, alpha = 1);  convert_element_type_default_11 = amax_default_1 = None
        return (mul_tensor, var_correction, sqrt_default, div_tensor, add_tensor_6, convert_element_type_default_10, sub_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
