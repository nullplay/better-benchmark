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
    def forward(self, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154, input_155, input_156, input_157, input_158, input_159, input_160, input_161, input_162, input_163, input_164, input_165, input_166, input_167, input_168, input_169, input_170, input_171, input_172, input_173, input_174, input_175, input_176, input_177, input_178, input_179, input_180, input_181, input_182, input_183, input_184, input_185, input_186, input_187, input_188, input_189, input_190, input_191, input_192, input_193, input_194, input_195, input_196, input_197, input_198, input_199, input_200, input_201, input_202, input_203, input_204, input_205, input_206, input_207, input_208, input_209):
        mul_tensor = torch.ops.aten.mul.Tensor(input_147, 0.5)
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_147, torch.float32);  input_147 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_148, 0.7978845608028654);  input_148 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_149, input_150);  input_149 = input_150 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_151, input_152, alpha = 1);  input_151 = input_152 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_2, alpha = 1);  sub_tensor = mul_tensor_2 = None
        div_tensor = torch.ops.aten.div.Tensor(input_153, 128);  input_153 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_1);  div_tensor = sub_tensor_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor);  mul_tensor = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor);  mul_tensor_3 = add_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, mul_tensor_6, alpha = 1);  mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, sub_tensor_2);  mul_tensor_4 = sub_tensor_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.7978845608028654);  mul_tensor_7 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 0.044715);  mul_tensor_8 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2.0);  convert_element_type_default = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_scalar);  mul_tensor_9 = mul_scalar = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_2, convert_element_type_default_3, alpha = 1);  convert_element_type_default_2 = convert_element_type_default_3 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.5);  convert_element_type_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_11, alpha = 1);  add_tensor_1 = mul_tensor_11 = None
        view_default = torch.ops.aten.view.default(add_tensor_2, input_154);  add_tensor_2 = input_154 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_155);  input_155 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_156);  permute_default = input_156 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_157);  mm_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_158);  input_158 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, 4096)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [2], True, dtype = None)
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_12, input_159);  mul_tensor_12 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [2], True, dtype = None);  mul_tensor_14 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(input_159, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_13, sum_dim_int_list, alpha = 1);  mul_tensor_13 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_15, alpha = 1);  sub_tensor_3 = mul_tensor_15 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(input_160, sub_tensor_4);  input_160 = sub_tensor_4 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_159);  input_159 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1], False, dtype = None);  mul_tensor_17 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 1], False, dtype = None);  convert_element_type_default_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16)
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_16, input_161, alpha = 1);  mul_tensor_16 = input_161 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(input_162, input_163, alpha = 1);  input_162 = input_163 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_164, torch.float32);  input_164 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_7, alpha = 1);  add_tensor_4 = convert_element_type_default_7 = None
        view_default_2 = torch.ops.aten.view.default(input_165, input_157);  input_165 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, convert_element_type_default_8, alpha = 1);  add_tensor_5 = convert_element_type_default_8 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(add_tensor_6, input_166);  input_166 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1], False, dtype = None);  mul_tensor_18 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(add_tensor_6, [0, 1], False, dtype = None);  add_tensor_6 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(sum_dim_int_list_2, sum_dim_int_list_4, alpha = 1);  sum_dim_int_list_2 = sum_dim_int_list_4 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(sum_dim_int_list_3, sum_dim_int_list_5, alpha = 1);  sum_dim_int_list_3 = sum_dim_int_list_5 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(input_167, input_168, alpha = 1);  input_167 = input_168 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_169, torch.float32);  input_169 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_9, convert_element_type_default_9, alpha = 1);  add_tensor_9 = convert_element_type_default_9 = None
        view_default_3 = torch.ops.aten.view.default(input_170, input_157);  input_170 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_10, convert_element_type_default_10, alpha = 1);  add_tensor_10 = convert_element_type_default_10 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(add_tensor_11, input_171);  input_171 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1], False, dtype = None);  mul_tensor_19 = None
        sum_dim_int_list_7 = torch.ops.aten.sum.dim_IntList(add_tensor_11, [0, 1], False, dtype = None);  add_tensor_11 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_7, sum_dim_int_list_6, alpha = 1);  add_tensor_7 = sum_dim_int_list_6 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_8, sum_dim_int_list_7, alpha = 1);  add_tensor_8 = sum_dim_int_list_7 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(input_172, input_173, alpha = 1);  input_172 = input_173 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(input_174, torch.float32);  input_174 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(add_tensor_14, convert_element_type_default_11, alpha = 1);  add_tensor_14 = convert_element_type_default_11 = None
        view_default_4 = torch.ops.aten.view.default(input_175, input_157);  input_175 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(add_tensor_15, convert_element_type_default_12, alpha = 1);  add_tensor_15 = convert_element_type_default_12 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(add_tensor_16, input_176);  input_176 = None
        sum_dim_int_list_8 = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1], False, dtype = None);  mul_tensor_20 = None
        sum_dim_int_list_9 = torch.ops.aten.sum.dim_IntList(add_tensor_16, [0, 1], False, dtype = None);  add_tensor_16 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_12, sum_dim_int_list_8, alpha = 1);  add_tensor_12 = sum_dim_int_list_8 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(add_tensor_13, sum_dim_int_list_9, alpha = 1);  add_tensor_13 = sum_dim_int_list_9 = None
        add_tensor_19 = torch.ops.aten.add.Tensor(input_177, input_178, alpha = 1);  input_177 = input_178 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(input_179, torch.float32);  input_179 = None
        add_tensor_20 = torch.ops.aten.add.Tensor(add_tensor_19, convert_element_type_default_13, alpha = 1);  add_tensor_19 = convert_element_type_default_13 = None
        view_default_5 = torch.ops.aten.view.default(input_180, input_157);  input_180 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor_21 = torch.ops.aten.add.Tensor(add_tensor_20, convert_element_type_default_14, alpha = 1);  add_tensor_20 = convert_element_type_default_14 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(add_tensor_21, input_181);  input_181 = None
        sum_dim_int_list_10 = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1], False, dtype = None);  mul_tensor_21 = None
        sum_dim_int_list_11 = torch.ops.aten.sum.dim_IntList(add_tensor_21, [0, 1], False, dtype = None);  add_tensor_21 = None
        add_tensor_22 = torch.ops.aten.add.Tensor(add_tensor_17, sum_dim_int_list_10, alpha = 1);  add_tensor_17 = sum_dim_int_list_10 = None
        add_tensor_23 = torch.ops.aten.add.Tensor(add_tensor_18, sum_dim_int_list_11, alpha = 1);  add_tensor_18 = sum_dim_int_list_11 = None
        add_tensor_24 = torch.ops.aten.add.Tensor(input_182, input_183, alpha = 1);  input_182 = input_183 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(input_184, torch.float32);  input_184 = None
        add_tensor_25 = torch.ops.aten.add.Tensor(add_tensor_24, convert_element_type_default_15, alpha = 1);  add_tensor_24 = convert_element_type_default_15 = None
        view_default_6 = torch.ops.aten.view.default(input_185, input_157);  input_185 = None
        convert_element_type_default_16 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        add_tensor_26 = torch.ops.aten.add.Tensor(add_tensor_25, convert_element_type_default_16, alpha = 1);  add_tensor_25 = convert_element_type_default_16 = None
        sum_dim_int_list_12 = torch.ops.aten.sum.dim_IntList(add_tensor_26, [0, 1], False, dtype = None);  add_tensor_26 = None
        add_tensor_27 = torch.ops.aten.add.Tensor(add_tensor_23, sum_dim_int_list_12, alpha = 1);  add_tensor_23 = sum_dim_int_list_12 = None
        add_tensor_28 = torch.ops.aten.add.Tensor(input_186, input_187, alpha = 1);  input_186 = input_187 = None
        convert_element_type_default_17 = torch.ops.prims.convert_element_type.default(input_188, torch.float32);  input_188 = None
        add_tensor_29 = torch.ops.aten.add.Tensor(add_tensor_28, convert_element_type_default_17, alpha = 1);  add_tensor_28 = convert_element_type_default_17 = None
        view_default_7 = torch.ops.aten.view.default(input_189, input_157);  input_189 = None
        convert_element_type_default_18 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor_30 = torch.ops.aten.add.Tensor(add_tensor_29, convert_element_type_default_18, alpha = 1);  add_tensor_29 = convert_element_type_default_18 = None
        sum_dim_int_list_13 = torch.ops.aten.sum.dim_IntList(add_tensor_30, [0, 1], False, dtype = None);  add_tensor_30 = None
        add_tensor_31 = torch.ops.aten.add.Tensor(add_tensor_27, sum_dim_int_list_13, alpha = 1);  add_tensor_27 = sum_dim_int_list_13 = None
        add_tensor_32 = torch.ops.aten.add.Tensor(input_190, input_191, alpha = 1);  input_190 = input_191 = None
        convert_element_type_default_19 = torch.ops.prims.convert_element_type.default(input_192, torch.float32);  input_192 = None
        add_tensor_33 = torch.ops.aten.add.Tensor(add_tensor_32, convert_element_type_default_19, alpha = 1);  add_tensor_32 = convert_element_type_default_19 = None
        view_default_8 = torch.ops.aten.view.default(input_193, input_157);  input_193 = None
        convert_element_type_default_20 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor_34 = torch.ops.aten.add.Tensor(add_tensor_33, convert_element_type_default_20, alpha = 1);  add_tensor_33 = convert_element_type_default_20 = None
        sum_dim_int_list_14 = torch.ops.aten.sum.dim_IntList(add_tensor_34, [0, 1], False, dtype = None);  add_tensor_34 = None
        add_tensor_35 = torch.ops.aten.add.Tensor(add_tensor_31, sum_dim_int_list_14, alpha = 1);  add_tensor_31 = sum_dim_int_list_14 = None
        add_tensor_36 = torch.ops.aten.add.Tensor(input_194, input_195, alpha = 1);  input_194 = input_195 = None
        convert_element_type_default_21 = torch.ops.prims.convert_element_type.default(input_196, torch.float32);  input_196 = None
        add_tensor_37 = torch.ops.aten.add.Tensor(add_tensor_36, convert_element_type_default_21, alpha = 1);  add_tensor_36 = convert_element_type_default_21 = None
        view_default_9 = torch.ops.aten.view.default(input_197, input_157);  input_197 = None
        convert_element_type_default_22 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        add_tensor_38 = torch.ops.aten.add.Tensor(add_tensor_37, convert_element_type_default_22, alpha = 1);  add_tensor_37 = convert_element_type_default_22 = None
        sum_dim_int_list_15 = torch.ops.aten.sum.dim_IntList(add_tensor_38, [0, 1], False, dtype = None);  add_tensor_38 = None
        add_tensor_39 = torch.ops.aten.add.Tensor(add_tensor_35, sum_dim_int_list_15, alpha = 1);  add_tensor_35 = sum_dim_int_list_15 = None
        add_tensor_40 = torch.ops.aten.add.Tensor(input_198, input_199, alpha = 1);  input_198 = input_199 = None
        convert_element_type_default_23 = torch.ops.prims.convert_element_type.default(input_200, torch.float32);  input_200 = None
        add_tensor_41 = torch.ops.aten.add.Tensor(add_tensor_40, convert_element_type_default_23, alpha = 1);  add_tensor_40 = convert_element_type_default_23 = None
        view_default_10 = torch.ops.aten.view.default(input_201, input_157);  input_201 = None
        convert_element_type_default_24 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_42 = torch.ops.aten.add.Tensor(add_tensor_41, convert_element_type_default_24, alpha = 1);  add_tensor_41 = convert_element_type_default_24 = None
        sum_dim_int_list_16 = torch.ops.aten.sum.dim_IntList(add_tensor_42, [0, 1], False, dtype = None);  add_tensor_42 = None
        add_tensor_43 = torch.ops.aten.add.Tensor(add_tensor_39, sum_dim_int_list_16, alpha = 1);  add_tensor_39 = sum_dim_int_list_16 = None
        add_tensor_44 = torch.ops.aten.add.Tensor(input_202, input_203, alpha = 1);  input_202 = input_203 = None
        convert_element_type_default_25 = torch.ops.prims.convert_element_type.default(input_204, torch.float32);  input_204 = None
        add_tensor_45 = torch.ops.aten.add.Tensor(add_tensor_44, convert_element_type_default_25, alpha = 1);  add_tensor_44 = convert_element_type_default_25 = None
        view_default_11 = torch.ops.aten.view.default(input_205, input_157);  input_205 = None
        convert_element_type_default_26 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32);  view_default_11 = None
        add_tensor_46 = torch.ops.aten.add.Tensor(add_tensor_45, convert_element_type_default_26, alpha = 1);  add_tensor_45 = convert_element_type_default_26 = None
        sum_dim_int_list_17 = torch.ops.aten.sum.dim_IntList(add_tensor_46, [0, 1], False, dtype = None);  add_tensor_46 = None
        add_tensor_47 = torch.ops.aten.add.Tensor(add_tensor_43, sum_dim_int_list_17, alpha = 1);  add_tensor_43 = sum_dim_int_list_17 = None
        add_tensor_48 = torch.ops.aten.add.Tensor(input_206, input_207, alpha = 1);  input_206 = input_207 = None
        convert_element_type_default_27 = torch.ops.prims.convert_element_type.default(input_208, torch.float32);  input_208 = None
        add_tensor_49 = torch.ops.aten.add.Tensor(add_tensor_48, convert_element_type_default_27, alpha = 1);  add_tensor_48 = convert_element_type_default_27 = None
        view_default_12 = torch.ops.aten.view.default(input_209, input_157);  input_209 = input_157 = None
        convert_element_type_default_28 = torch.ops.prims.convert_element_type.default(view_default_12, torch.float32);  view_default_12 = None
        add_tensor_50 = torch.ops.aten.add.Tensor(add_tensor_49, convert_element_type_default_28, alpha = 1);  add_tensor_49 = convert_element_type_default_28 = None
        sum_dim_int_list_18 = torch.ops.aten.sum.dim_IntList(add_tensor_50, [0, 1], False, dtype = None);  add_tensor_50 = None
        add_tensor_51 = torch.ops.aten.add.Tensor(add_tensor_47, sum_dim_int_list_18, alpha = 1);  add_tensor_47 = sum_dim_int_list_18 = None
        return (convert_element_type_default_5, convert_element_type_default_6, add_tensor_3, add_tensor_22, add_tensor_51)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
