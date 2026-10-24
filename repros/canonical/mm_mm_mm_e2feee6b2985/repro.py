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
    def forward(self, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154, input_155, input_156, input_157, input_158, input_159, input_160, input_161, input_162, input_163, input_164, input_165, input_166, input_167, input_168, input_169, input_170, input_171, input_172, input_173, input_174, input_175, input_176, input_177, input_178, input_179, input_180, input_181, input_182):
        mul_tensor = torch.ops.aten.mul.Tensor(input_125, input_126);  input_125 = input_126 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_127, torch.float32);  input_127 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default, alpha = 1);  mul_tensor = convert_element_type_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], False, dtype = None);  add_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_128, input_129);  input_128 = input_129 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_130, torch.float32);  input_130 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, convert_element_type_default_1, alpha = 1);  mul_tensor_1 = convert_element_type_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_131, input_132, alpha = 1);  input_131 = input_132 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], False, dtype = None);  add_tensor_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(sum_dim_int_list, sum_dim_int_list_1, alpha = 1);  sum_dim_int_list = sum_dim_int_list_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_133, input_134);  input_133 = input_134 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_135, torch.float32);  input_135 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_2, convert_element_type_default_2, alpha = 1);  mul_tensor_2 = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_136, torch.float32);  input_136 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_3, alpha = 1);  add_tensor_2 = convert_element_type_default_3 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(input_137, input_138, alpha = 1);  input_137 = input_138 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0, 1], False, dtype = None);  add_tensor_4 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(input_139, input_140, alpha = 1);  input_139 = input_140 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_3, sum_dim_int_list_2, alpha = 1);  add_tensor_3 = sum_dim_int_list_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_141, input_142);  input_141 = input_142 = None
        mm_default = torch.ops.aten.mm.default(input_143, input_144);  input_143 = input_144 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_145, torch.float32);  input_145 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(mul_tensor_3, convert_element_type_default_4, alpha = 1);  mul_tensor_3 = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_5, convert_element_type_default_5, alpha = 1);  add_tensor_5 = convert_element_type_default_5 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_146, torch.float32);  input_146 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_6, convert_element_type_default_6, alpha = 1);  add_tensor_6 = convert_element_type_default_6 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_9, input_147);  input_147 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], False, dtype = None);  mul_tensor_4 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(add_tensor_9, [0, 1], False, dtype = None);  add_tensor_9 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_3, alpha = 1);  add_tensor_7 = sum_dim_int_list_3 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_8, sum_dim_int_list_4, alpha = 1);  add_tensor_8 = sum_dim_int_list_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_148, input_149);  input_148 = input_149 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 4096)
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_150);  mul_tensor_5 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True, dtype = None);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_150, sum_dim_int_list_6);  input_150 = sum_dim_int_list_6 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_6, sum_dim_int_list_5, alpha = 1);  mul_tensor_6 = sum_dim_int_list_5 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_8, alpha = 1);  sub_tensor = mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(input_151, sub_tensor_1);  input_151 = sub_tensor_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default_7, input_152);  convert_element_type_default_7 = input_152 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_153);  input_153 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_154);  permute_default = input_154 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_155);  mm_default_1 = input_155 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_156, 0.5)
        mul_tensor_11 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, mul_tensor_10);  mul_tensor_10 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_156, torch.float32);  input_156 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_157, 0.7978845608028654);  input_157 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_12);  mul_tensor_12 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_13 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, add_tensor_14);  convert_element_type_default_8 = add_tensor_14 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(mul_tensor_13, torch.bfloat16);  mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, mul_tensor_14, alpha = 1);  mul_tensor_14 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_11, sub_tensor_2);  mul_tensor_11 = sub_tensor_2 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_15, 0.7978845608028654);  mul_tensor_15 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16)
        mul_tensor_17 = torch.ops.aten.mul.Tensor(mul_tensor_16, 0.044715);  mul_tensor_16 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_9, 2.0);  convert_element_type_default_9 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_17, mul_scalar);  mul_tensor_17 = mul_scalar = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(mul_tensor_18, torch.bfloat16);  mul_tensor_18 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(convert_element_type_default_11, convert_element_type_default_12, alpha = 1);  convert_element_type_default_11 = convert_element_type_default_12 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(convert_element_type_default_10, 0.5);  convert_element_type_default_10 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(add_tensor_15, mul_tensor_19, alpha = 1);  add_tensor_15 = mul_tensor_19 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_16, input_158);  add_tensor_16 = input_158 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_2, input_159);  input_159 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_1, input_160);  permute_default_1 = input_160 = None
        sum_dim_int_list_7 = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True, dtype = torch.float32);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list_7, input_161);  sum_dim_int_list_7 = input_161 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(view_default_3, torch.bfloat16);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_3, input_162);  mm_default_3 = input_162 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(mul_tensor_9, convert_element_type_default_14, alpha = 1);  mul_tensor_9 = convert_element_type_default_14 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(add_tensor_10, convert_element_type_default_15, alpha = 1);  add_tensor_10 = convert_element_type_default_15 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(convert_element_type_default_13, torch.float32);  convert_element_type_default_13 = None
        add_tensor_19 = torch.ops.aten.add.Tensor(add_tensor_11, convert_element_type_default_16, alpha = 1);  add_tensor_11 = convert_element_type_default_16 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(add_tensor_17, input_163);  input_163 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(mul_tensor_20, 4096)
        sum_dim_int_list_8 = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [2], True, dtype = None)
        mul_tensor_22 = torch.ops.aten.mul.Tensor(mul_tensor_20, input_164);  mul_tensor_20 = None
        sum_dim_int_list_9 = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [2], True, dtype = None);  mul_tensor_22 = None
        mul_tensor_23 = torch.ops.aten.mul.Tensor(input_164, sum_dim_int_list_9);  sum_dim_int_list_9 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_21, sum_dim_int_list_8, alpha = 1);  mul_tensor_21 = sum_dim_int_list_8 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_23, alpha = 1);  sub_tensor_3 = mul_tensor_23 = None
        mul_tensor_24 = torch.ops.aten.mul.Tensor(input_165, sub_tensor_4);  input_165 = sub_tensor_4 = None
        mul_tensor_25 = torch.ops.aten.mul.Tensor(add_tensor_17, input_164);  input_164 = None
        sum_dim_int_list_10 = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1], False, dtype = None);  mul_tensor_25 = None
        sum_dim_int_list_11 = torch.ops.aten.sum.dim_IntList(add_tensor_17, [0, 1], False, dtype = None);  add_tensor_17 = None
        add_tensor_20 = torch.ops.aten.add.Tensor(add_tensor_12, sum_dim_int_list_10, alpha = 1);  add_tensor_12 = sum_dim_int_list_10 = None
        add_tensor_21 = torch.ops.aten.add.Tensor(add_tensor_13, sum_dim_int_list_11, alpha = 1);  add_tensor_13 = sum_dim_int_list_11 = None
        mul_tensor_26 = torch.ops.aten.mul.Tensor(input_166, input_167);  input_166 = input_167 = None
        mm_default_5 = torch.ops.aten.mm.default(input_168, input_169);  input_168 = input_169 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(input_170, torch.float32);  input_170 = None
        add_tensor_22 = torch.ops.aten.add.Tensor(mul_tensor_26, convert_element_type_default_17, alpha = 1);  mul_tensor_26 = convert_element_type_default_17 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        add_tensor_23 = torch.ops.aten.add.Tensor(add_tensor_18, convert_element_type_default_18, alpha = 1);  add_tensor_18 = convert_element_type_default_18 = None
        convert_element_type_default_19 = torch.ops.prims.convert_element_type.default(input_171, torch.float32);  input_171 = None
        add_tensor_24 = torch.ops.aten.add.Tensor(add_tensor_19, convert_element_type_default_19, alpha = 1);  add_tensor_19 = convert_element_type_default_19 = None
        mul_tensor_27 = torch.ops.aten.mul.Tensor(add_tensor_22, input_172);  input_172 = None
        sum_dim_int_list_12 = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1], False, dtype = None);  mul_tensor_27 = None
        sum_dim_int_list_13 = torch.ops.aten.sum.dim_IntList(add_tensor_22, [0, 1], False, dtype = None);  add_tensor_22 = None
        add_tensor_25 = torch.ops.aten.add.Tensor(add_tensor_20, sum_dim_int_list_12, alpha = 1);  add_tensor_20 = sum_dim_int_list_12 = None
        add_tensor_26 = torch.ops.aten.add.Tensor(add_tensor_21, sum_dim_int_list_13, alpha = 1);  add_tensor_21 = sum_dim_int_list_13 = None
        mul_tensor_28 = torch.ops.aten.mul.Tensor(input_173, input_174);  input_173 = input_174 = None
        convert_element_type_default_20 = torch.ops.prims.convert_element_type.default(input_175, torch.float32);  input_175 = None
        add_tensor_27 = torch.ops.aten.add.Tensor(mul_tensor_28, convert_element_type_default_20, alpha = 1);  mul_tensor_28 = convert_element_type_default_20 = None
        convert_element_type_default_21 = torch.ops.prims.convert_element_type.default(input_176, torch.float32);  input_176 = None
        add_tensor_28 = torch.ops.aten.add.Tensor(add_tensor_23, convert_element_type_default_21, alpha = 1);  add_tensor_23 = convert_element_type_default_21 = None
        add_tensor_29 = torch.ops.aten.add.Tensor(add_tensor_24, input_177, alpha = 1);  add_tensor_24 = input_177 = None
        sum_dim_int_list_14 = torch.ops.aten.sum.dim_IntList(add_tensor_27, [0, 1], False, dtype = None);  add_tensor_27 = None
        add_tensor_30 = torch.ops.aten.add.Tensor(add_tensor_25, input_178, alpha = 1);  add_tensor_25 = input_178 = None
        add_tensor_31 = torch.ops.aten.add.Tensor(add_tensor_26, sum_dim_int_list_14, alpha = 1);  add_tensor_26 = sum_dim_int_list_14 = None
        mul_tensor_29 = torch.ops.aten.mul.Tensor(input_179, input_180);  input_179 = input_180 = None
        convert_element_type_default_22 = torch.ops.prims.convert_element_type.default(input_181, torch.float32);  input_181 = None
        add_tensor_32 = torch.ops.aten.add.Tensor(mul_tensor_29, convert_element_type_default_22, alpha = 1);  mul_tensor_29 = convert_element_type_default_22 = None
        add_tensor_33 = torch.ops.aten.add.Tensor(add_tensor_28, input_182, alpha = 1);  add_tensor_28 = input_182 = None
        sum_dim_int_list_15 = torch.ops.aten.sum.dim_IntList(add_tensor_32, [0, 1], False, dtype = None);  add_tensor_32 = None
        add_tensor_34 = torch.ops.aten.add.Tensor(add_tensor_31, sum_dim_int_list_15, alpha = 1);  add_tensor_31 = sum_dim_int_list_15 = None
        return (mm_default_2, mul_tensor_24, add_tensor_29, add_tensor_30, add_tensor_33, add_tensor_34)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
