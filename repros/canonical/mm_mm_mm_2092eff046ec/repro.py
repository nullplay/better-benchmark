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
    def forward(self, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118):
        mm_default = torch.ops.aten.mm.default(input_86, input_87);  input_86 = input_87 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        add_tensor = torch.ops.aten.add.Tensor(input_88, input_89, alpha = 1);  input_88 = input_89 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], False, dtype = None);  add_tensor = None
        mm_default_1 = torch.ops.aten.mm.default(input_90, input_91);  input_90 = input_91 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_1, alpha = 1);  convert_element_type_default = convert_element_type_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_92, input_93, alpha = 1);  input_92 = input_93 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], False, dtype = None);  add_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_94, input_95, alpha = 1);  input_94 = input_95 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(sum_dim_int_list, sum_dim_int_list_1, alpha = 1);  sum_dim_int_list = sum_dim_int_list_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(input_96, input_97, alpha = 1);  input_96 = input_97 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_5, input_98);  add_tensor_5 = input_98 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 4096)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_99);  mul_tensor = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_99, sum_dim_int_list_3);  input_99 = sum_dim_int_list_3 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list_2, alpha = 1);  mul_tensor_1 = sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_100, sub_tensor_1);  input_100 = sub_tensor_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_101);  convert_element_type_default_2 = input_101 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default, input_102);  input_102 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default, input_103);  permute_default = input_103 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_2, input_104);  mm_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_4, alpha = 1);  add_tensor_1 = convert_element_type_default_4 = None
        view_default_2 = torch.ops.aten.view.default(input_105, input_104);  input_105 = input_104 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_5);  mul_tensor_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_5, 3.0)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_7 = torch.ops.aten.add.Tensor(view_default_2, mul_tensor_7, alpha = 1);  view_default_2 = mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_tensor_7, 0.7978845608028654);  add_tensor_7 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_8);  mul_tensor_8 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, add_tensor_8);  convert_element_type_default_3 = add_tensor_8 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, mul_tensor_10, alpha = 1);  mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_6, sub_tensor_2);  mul_tensor_6 = sub_tensor_2 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, 0.7978845608028654);  mul_tensor_11 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16)
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.044715);  mul_tensor_12 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_5, 2.0);  convert_element_type_default_5 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 3.0);  pow_tensor_scalar_1 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, mul_scalar);  mul_tensor_13 = mul_scalar = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mul_tensor_14, torch.bfloat16);  mul_tensor_14 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(convert_element_type_default_7, convert_element_type_default_8, alpha = 1);  convert_element_type_default_7 = convert_element_type_default_8 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 0.5);  convert_element_type_default_6 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_9, mul_tensor_15, alpha = 1);  add_tensor_9 = mul_tensor_15 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_10, input_106);  add_tensor_10 = input_106 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_3, input_107);  input_107 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 0])
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = torch.float32);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_4, input_108);  mm_default_4 = input_108 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(mul_tensor_4, convert_element_type_default_9, alpha = 1);  mul_tensor_4 = convert_element_type_default_9 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(add_tensor_11, input_109);  input_109 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(mul_tensor_16, 4096)
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [2], True, dtype = None)
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_16, input_110);  mul_tensor_16 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [2], True, dtype = None);  mul_tensor_18 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(input_110, sum_dim_int_list_6);  sum_dim_int_list_6 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_17, sum_dim_int_list_5, alpha = 1);  mul_tensor_17 = sum_dim_int_list_5 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_19, alpha = 1);  sub_tensor_3 = mul_tensor_19 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(add_tensor_11, input_110);  input_110 = None
        sum_dim_int_list_7 = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1], False, dtype = None);  mul_tensor_20 = None
        sum_dim_int_list_8 = torch.ops.aten.sum.dim_IntList(add_tensor_11, [0, 1], False, dtype = None);  add_tensor_11 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_3, sum_dim_int_list_7, alpha = 1);  add_tensor_3 = sum_dim_int_list_7 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_4, sum_dim_int_list_8, alpha = 1);  add_tensor_4 = sum_dim_int_list_8 = None
        mm_default_5 = torch.ops.aten.mm.default(input_111, input_112);  input_111 = input_112 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(add_tensor_6, convert_element_type_default_10, alpha = 1);  add_tensor_6 = convert_element_type_default_10 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(input_113, input_114, alpha = 1);  input_113 = input_114 = None
        sum_dim_int_list_9 = torch.ops.aten.sum.dim_IntList(add_tensor_15, [0, 1], False, dtype = None);  add_tensor_15 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(add_tensor_12, input_115, alpha = 1);  add_tensor_12 = input_115 = None
        add_tensor_17 = torch.ops.aten.add.Tensor(add_tensor_13, sum_dim_int_list_9, alpha = 1);  add_tensor_13 = sum_dim_int_list_9 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(input_116, torch.float32);  input_116 = None
        add_tensor_18 = torch.ops.aten.add.Tensor(add_tensor_14, convert_element_type_default_11, alpha = 1);  add_tensor_14 = convert_element_type_default_11 = None
        add_tensor_19 = torch.ops.aten.add.Tensor(add_tensor_17, input_117, alpha = 1);  add_tensor_17 = input_117 = None
        add_tensor_20 = torch.ops.aten.add.Tensor(add_tensor_18, input_118, alpha = 1);  add_tensor_18 = input_118 = None
        return (permute_default_1, sum_dim_int_list_4, sub_tensor_4, add_tensor_16, add_tensor_19, add_tensor_20)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
