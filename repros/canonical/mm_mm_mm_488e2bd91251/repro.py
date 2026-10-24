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
    def forward(self, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65):
        mul_tensor = torch.ops.aten.mul.Tensor(input_48, input_49);  input_48 = input_49 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_50, input_51, alpha = 1);  input_50 = input_51 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor, alpha = 1);  sub_tensor = mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_52, sub_tensor_1);  input_52 = sub_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_53, 1.1111111111111112);  input_53 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_2);  mul_tensor_2 = None
        view_default = torch.ops.aten.view.default(mul_tensor_3, input_54);  mul_tensor_3 = input_54 = None
        permute_default = torch.ops.aten.permute.default(input_55, [1, 0]);  input_55 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_56);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(input_57, input_56);  input_57 = input_56 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_1, mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_58, 0.7978845608028654);  input_58 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_6);  mul_tensor_6 = None
        add_tensor = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_1, add_tensor);  view_default_1 = add_tensor = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, mul_tensor_8, alpha = 1);  mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_5, sub_tensor_2);  mul_tensor_5 = sub_tensor_2 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, 0.7978845608028654);  mul_tensor_9 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.044715)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(view_default_2, 2.0);  view_default_2 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_11, mul_scalar);  mul_tensor_11 = mul_scalar = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_10, mul_tensor_12, alpha = 1);  mul_tensor_10 = mul_tensor_12 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.5);  mul_tensor_7 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_13, alpha = 1);  add_tensor_1 = mul_tensor_13 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_2, input_59);  add_tensor_2 = input_59 = None
        permute_default_1 = torch.ops.aten.permute.default(input_60, [1, 0]);  input_60 = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default_1, [1, 0]);  permute_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, permute_default_2);  permute_default_2 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default_3, input_61);  permute_default_3 = input_61 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = None);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_62);  sum_dim_int_list = input_62 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_63);  mm_default_1 = input_63 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, view_default_5, alpha = 1);  mul_tensor_1 = view_default_5 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(add_tensor_3, input_64);  input_64 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_14, 768)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [2], True, dtype = None)
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_14, input_65);  mul_tensor_14 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [2], True, dtype = None);  mul_tensor_16 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(input_65, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_15, sum_dim_int_list_1, alpha = 1);  mul_tensor_15 = sum_dim_int_list_1 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(add_tensor_3, input_65);  input_65 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1], False, dtype = None);  mul_tensor_18 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0, 1], False, dtype = None);  add_tensor_3 = None
        return (mm_default_2, view_default_4, mul_tensor_17, sub_tensor_3, sum_dim_int_list_3, sum_dim_int_list_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
