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
    def forward(self, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53):
        view_default = torch.ops.aten.view.default(input_41, input_42);  input_41 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default, 0.5)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_43, 0.7978845608028654);  input_43 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_1);  mul_tensor_1 = None
        add_tensor = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_44, input_45);  input_44 = input_45 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_46, input_47, alpha = 1);  input_46 = input_47 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_2, alpha = 1);  sub_tensor = mul_tensor_2 = None
        div_tensor = torch.ops.aten.div.Tensor(input_48, 768);  input_48 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_1);  div_tensor = sub_tensor_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor);  mul_tensor = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor);  mul_tensor_3 = add_tensor = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, mul_tensor_6, alpha = 1);  mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, sub_tensor_2);  mul_tensor_4 = sub_tensor_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 0.7978845608028654);  mul_tensor_7 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, 0.044715)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(view_default, 2.0);  view_default = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_scalar);  mul_tensor_9 = mul_scalar = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_10, alpha = 1);  mul_tensor_8 = mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_5, 0.5);  mul_tensor_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_11, alpha = 1);  add_tensor_1 = mul_tensor_11 = None
        view_default_1 = torch.ops.aten.view.default(add_tensor_2, input_49);  add_tensor_2 = input_49 = None
        permute_default = torch.ops.aten.permute.default(input_50, [1, 0]);  input_50 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default_1);  permute_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_51);  permute_default_2 = input_51 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_42);  mm_default = input_42 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(view_default_2, input_52);  input_52 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [2], True, dtype = None)
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_12, input_53);  mul_tensor_12 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [2], True, dtype = None);  mul_tensor_14 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(input_53, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_13, sum_dim_int_list, alpha = 1);  mul_tensor_13 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_15, alpha = 1);  sub_tensor_3 = mul_tensor_15 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(view_default_2, input_53);  input_53 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1], False, dtype = None);  mul_tensor_16 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_2, [0, 1], False, dtype = None);  view_default_2 = None
        return (mm_default_1, sub_tensor_4, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
