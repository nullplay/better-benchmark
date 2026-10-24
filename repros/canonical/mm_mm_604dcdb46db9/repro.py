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
    def forward(self, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73):
        mul_tensor = torch.ops.aten.mul.Tensor(input_57, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_57, [2], True, dtype = None);  input_57 = None
        view_default = torch.ops.aten.view.default(input_58, input_59);  input_58 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default, 0.5)
        add_tensor = torch.ops.aten.add.Tensor(view_default, input_60, alpha = 1);  input_60 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_1)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_3, input_61, alpha = 1);  mul_tensor_3 = input_61 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor, input_62);  sub_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_63, [2], True, dtype = None);  input_63 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_1);  mul_tensor_4 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor, sum_dim_int_list, alpha = 1);  mul_tensor = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5, alpha = 1);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor = torch.ops.aten.div.Tensor(input_62, 768);  input_62 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_1);  mul_tensor_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_6, add_tensor_1);  mul_tensor_6 = add_tensor_1 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(1, mul_tensor_9, alpha = 1);  mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_7, sub_tensor_3);  mul_tensor_7 = sub_tensor_3 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.7978845608028654);  mul_tensor_10 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, 0.044715)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(view_default, 2.0);  view_default = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, mul_scalar);  mul_tensor_12 = mul_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_11, mul_tensor_13, alpha = 1);  mul_tensor_11 = mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_8, 0.5);  mul_tensor_8 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_14, alpha = 1);  add_tensor_2 = mul_tensor_14 = None
        view_default_1 = torch.ops.aten.view.default(add_tensor_3, input_64);  add_tensor_3 = None
        permute_default = torch.ops.aten.permute.default(input_65, [1, 0]);  input_65 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default_1);  permute_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_66);  permute_default_2 = input_66 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_59);  mm_default = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(view_default_2, input_67);  input_67 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_15, 768)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [2], True, dtype = None)
        mul_tensor_17 = torch.ops.aten.mul.Tensor(mul_tensor_15, input_68);  mul_tensor_15 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [2], True, dtype = None);  mul_tensor_17 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(input_68, sum_dim_int_list_3);  sum_dim_int_list_3 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_16, sum_dim_int_list_2, alpha = 1);  mul_tensor_16 = sum_dim_int_list_2 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_18, alpha = 1);  sub_tensor_4 = mul_tensor_18 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(input_69, sub_tensor_5);  input_69 = sub_tensor_5 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(view_default_2, input_68);  input_68 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1], False, dtype = None);  mul_tensor_20 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(view_default_2, [0, 1], False, dtype = None);  view_default_2 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(input_70, 1.1111111111111112);  input_70 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(mul_tensor_19, mul_tensor_21);  mul_tensor_21 = None
        view_default_3 = torch.ops.aten.view.default(mul_tensor_22, input_64);  mul_tensor_22 = input_64 = None
        view_default_4 = torch.ops.aten.view.default(input_71, input_59);  input_71 = input_59 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_19, view_default_4, alpha = 1);  mul_tensor_19 = view_default_4 = None
        mul_tensor_23 = torch.ops.aten.mul.Tensor(add_tensor_4, input_72);  input_72 = None
        mul_tensor_24 = torch.ops.aten.mul.Tensor(add_tensor_4, input_73);  input_73 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0, 1], False, dtype = None);  add_tensor_4 = None
        return (mm_default_1, sum_dim_int_list_4, sum_dim_int_list_5, view_default_3, mul_tensor_23, mul_tensor_24, sum_dim_int_list_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
