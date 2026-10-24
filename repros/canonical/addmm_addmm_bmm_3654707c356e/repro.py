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
    def forward(self, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137, input_138, input_139, input_140, input_141, input_142, input_143, input_144, input_145, input_146, input_147, input_148, input_149, input_150, input_151, input_152, input_153, input_154, input_155, input_156, input_157, input_158, input_159, input_160, input_161, input_162, input_163, input_164, input_165, input_166, input_167, input_168, input_169, input_170, input_171, input_172, input_173, input_174, input_175, input_176, input_177, input_178, input_179, input_180, input_181, input_182, input_183, input_184, input_185, input_186):
        add_tensor = torch.ops.aten.add.Tensor(input_128, input_129, alpha = 1);  input_128 = input_129 = None
        view_default = torch.ops.aten.view.default(input_130, input_131);  input_130 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        addmm_default = torch.ops.aten.addmm.default(input_132, input_133, input_134, beta = 1, alpha = 1);  input_132 = input_133 = input_134 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_131);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_135);  convert_element_type_default = input_135 = None
        view_default_2 = torch.ops.aten.view.default(input_136, input_137);  input_136 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_2, [1], False, dtype = None);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list, input_131);  sum_dim_int_list = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(input_138);  input_138 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_default);  convert_element_type_default_1 = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_139, mul_tensor_1);  input_139 = mul_tensor_1 = None
        view_default_4 = torch.ops.aten.view.default(input_140, input_141);  input_140 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_142);  view_default_4 = None
        permute_default = torch.ops.aten.permute.default(input_143, [1, 0]);  input_143 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_144, view_default_5, permute_default, beta = 1, alpha = 1);  input_144 = view_default_5 = permute_default = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_131);  addmm_default_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_6, alpha = 1);  add_tensor_3 = view_default_6 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_2, [-1], True, dtype = None);  pow_tensor_scalar_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mean_dim_1, 1e-05, alpha = 1);  mean_dim_1 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default_1);  convert_element_type_default_2 = rsqrt_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_145, mul_tensor_3);  input_145 = mul_tensor_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        where_self = torch.ops.aten.where.self(input_146, input_147, input_148);  input_146 = input_147 = input_148 = None
        index_put_default = torch.ops.aten.index_put.default(input_149, [input_150], input_151, False);  input_149 = input_150 = input_151 = None
        index_tensor = torch.ops.aten.index.Tensor(where_self, [index_put_default]);  where_self = index_put_default = None
        view_default_7 = torch.ops.aten.view.default(index_tensor, input_137);  index_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_7, [1], False, dtype = None);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(sum_dim_int_list_1, input_131);  sum_dim_int_list_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_4, view_default_8, alpha = 1);  add_tensor_4 = view_default_8 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        pow_tensor_scalar_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2)
        mean_dim_2 = torch.ops.aten.mean.dim(pow_tensor_scalar_3, [-1], True, dtype = None);  pow_tensor_scalar_3 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mean_dim_2, 1e-05, alpha = 1);  mean_dim_2 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, rsqrt_default_2);  convert_element_type_default_4 = rsqrt_default_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_152, mul_tensor_5);  input_152 = mul_tensor_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_5, input_153)
        view_default_10 = torch.ops.aten.view.default(convert_element_type_default_5, input_153)
        expand_default = torch.ops.aten.expand.default(input_154, input_155, implicit = False);  input_154 = input_155 = None
        view_default_11 = torch.ops.aten.view.default(expand_default, input_156);  expand_default = input_156 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_5, input_153);  convert_element_type_default_5 = None
        expand_default_1 = torch.ops.aten.expand.default(input_157, input_158, implicit = False);  input_157 = None
        view_default_13 = torch.ops.aten.view.default(expand_default_1, input_159);  expand_default_1 = input_159 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_11, view_default_13);  view_default_11 = view_default_13 = None
        view_default_14 = torch.ops.aten.view.default(bmm_default, input_158);  bmm_default = input_158 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_14, [0, 2, 1, 3]);  view_default_14 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_15 = torch.ops.aten.view.default(clone_default, input_141);  clone_default = None
        view_default_16 = torch.ops.aten.view.default(view_default_15, input_142);  view_default_15 = None
        permute_default_2 = torch.ops.aten.permute.default(input_160, [1, 0]);  input_160 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_161, view_default_16, permute_default_2, beta = 1, alpha = 1);  input_161 = view_default_16 = permute_default_2 = None
        view_default_17 = torch.ops.aten.view.default(addmm_default_2, input_131);  addmm_default_2 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_6, view_default_17, alpha = 1);  add_tensor_6 = view_default_17 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.float32)
        pow_tensor_scalar_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_6, 2)
        mean_dim_3 = torch.ops.aten.mean.dim(pow_tensor_scalar_4, [-1], True, dtype = None);  pow_tensor_scalar_4 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(mean_dim_3, 1e-05, alpha = 1);  mean_dim_3 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, rsqrt_default_3);  convert_element_type_default_6 = rsqrt_default_3 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_162, mul_tensor_7);  input_162 = mul_tensor_7 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        view_default_18 = torch.ops.aten.view.default(convert_element_type_default_7, input_163);  convert_element_type_default_7 = input_163 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_164, view_default_18, input_165, beta = 1, alpha = 1);  input_164 = input_165 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(view_default_18, [input_166]);  view_default_18 = input_166 = None
        where_self_1 = torch.ops.aten.where.self(input_167, input_168, input_169);  input_167 = input_168 = input_169 = None
        index_put_default_1 = torch.ops.aten.index_put.default(input_170, [input_171], input_172, False);  input_170 = input_171 = input_172 = None
        index_tensor_2 = torch.ops.aten.index.Tensor(where_self_1, [index_put_default_1]);  where_self_1 = index_put_default_1 = None
        view_default_19 = torch.ops.aten.view.default(index_tensor_2, input_137);  index_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_19, [1], False, dtype = None);  view_default_19 = None
        view_default_20 = torch.ops.aten.view.default(sum_dim_int_list_2, input_131);  sum_dim_int_list_2 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_8, view_default_20, alpha = 1);  add_tensor_8 = view_default_20 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.float32)
        pow_tensor_scalar_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_8, 2)
        mean_dim_4 = torch.ops.aten.mean.dim(pow_tensor_scalar_5, [-1], True, dtype = None);  pow_tensor_scalar_5 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(mean_dim_4, 1e-05, alpha = 1);  mean_dim_4 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_11);  add_tensor_11 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, rsqrt_default_4);  convert_element_type_default_8 = rsqrt_default_4 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_173, mul_tensor_9);  input_173 = mul_tensor_9 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        view_default_21 = torch.ops.aten.view.default(convert_element_type_default_9, input_153)
        view_default_22 = torch.ops.aten.view.default(convert_element_type_default_9, input_153)
        view_default_23 = torch.ops.aten.view.default(convert_element_type_default_9, input_153);  convert_element_type_default_9 = input_153 = None
        view_default_24 = torch.ops.aten.view.default(input_174, input_141);  input_174 = input_141 = None
        view_default_25 = torch.ops.aten.view.default(view_default_24, input_142);  view_default_24 = input_142 = None
        permute_default_3 = torch.ops.aten.permute.default(input_175, [1, 0]);  input_175 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_176, view_default_25, permute_default_3, beta = 1, alpha = 1);  input_176 = view_default_25 = permute_default_3 = None
        view_default_26 = torch.ops.aten.view.default(addmm_default_4, input_131);  addmm_default_4 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_10, view_default_26, alpha = 1);  add_tensor_10 = view_default_26 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(add_tensor_12, torch.float32)
        pow_tensor_scalar_6 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_10, 2)
        mean_dim_5 = torch.ops.aten.mean.dim(pow_tensor_scalar_6, [-1], True, dtype = None);  pow_tensor_scalar_6 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(mean_dim_5, 1e-05, alpha = 1);  mean_dim_5 = None
        rsqrt_default_5 = torch.ops.aten.rsqrt.default(add_tensor_13);  add_tensor_13 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(convert_element_type_default_10, rsqrt_default_5);  convert_element_type_default_10 = rsqrt_default_5 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_177, mul_tensor_11);  input_177 = mul_tensor_11 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        view_default_27 = torch.ops.aten.view.default(input_178, input_137);  input_178 = input_137 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_27, [1], False, dtype = None);  view_default_27 = None
        view_default_28 = torch.ops.aten.view.default(sum_dim_int_list_3, input_131);  sum_dim_int_list_3 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(add_tensor_12, view_default_28, alpha = 1);  add_tensor_12 = view_default_28 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(add_tensor_14, torch.float32)
        pow_tensor_scalar_7 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_12, 2)
        mean_dim_6 = torch.ops.aten.mean.dim(pow_tensor_scalar_7, [-1], True, dtype = None);  pow_tensor_scalar_7 = None
        rsqrt_default_6 = torch.ops.aten.rsqrt.default(input_179);  input_179 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(convert_element_type_default_12, rsqrt_default_6);  convert_element_type_default_12 = rsqrt_default_6 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_180, mul_tensor_13);  input_180 = mul_tensor_13 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_181, input_182, input_183, beta = 1, alpha = 1);  input_181 = input_182 = input_183 = None
        view_default_29 = torch.ops.aten.view.default(addmm_default_5, input_131);  addmm_default_5 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(add_tensor_14, view_default_29, alpha = 1);  add_tensor_14 = view_default_29 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(add_tensor_15, torch.float32)
        pow_tensor_scalar_8 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_13, 2)
        mul_tensor_15 = torch.ops.aten.mul.Tensor(convert_element_type_default_13, input_184);  convert_element_type_default_13 = input_184 = None
        view_default_30 = torch.ops.aten.view.default(input_185, input_131);  input_185 = input_131 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(add_tensor_15, view_default_30, alpha = 1);  add_tensor_15 = view_default_30 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(add_tensor_16, torch.float32)
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_16, input_186, alpha = 1);  add_tensor_16 = input_186 = None
        return (pow_tensor_scalar, mul_tensor, mean_dim, mul_tensor_2, convert_element_type_default_3, view_default_9, view_default_10, view_default_12, addmm_default_3, index_tensor_1, view_default_21, view_default_22, view_default_23, convert_element_type_default_11, mean_dim_6, mul_tensor_14, pow_tensor_scalar_8, mul_tensor_15, convert_element_type_default_14, add_tensor_17)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
