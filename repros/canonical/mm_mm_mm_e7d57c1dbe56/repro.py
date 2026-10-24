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
    def forward(self, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66):
        mul_tensor = torch.ops.aten.mul.Tensor(input_50, input_51);  input_50 = input_51 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_52);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_52, sum_dim_int_list_1);  input_52 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_53, sub_tensor_1);  input_53 = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_54, torch.float32);  input_54 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_5 = None
        view_default = torch.ops.aten.view.default(mul_tensor_6, input_55);  mul_tensor_6 = input_55 = None
        permute_default = torch.ops.aten.permute.default(input_56, [1, 0]);  input_56 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default_1);  permute_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_57);  permute_default_2 = input_57 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_58);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(input_59, input_58);  input_59 = input_58 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_2, 0.5)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_default_1, mul_tensor_7);  mul_tensor_7 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_2, input_60, alpha = 1);  view_default_2 = input_60 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_9);  mul_tensor_9 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_default_1, add_tensor_1);  view_default_1 = add_tensor_1 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, mul_tensor_11, alpha = 1);  mul_tensor_11 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_8, sub_tensor_2);  mul_tensor_8 = sub_tensor_2 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.7978845608028654);  mul_tensor_12 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, 0.044715)
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_13, input_61, alpha = 1);  mul_tensor_13 = input_61 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.5);  mul_tensor_10 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_15, alpha = 1);  add_tensor_2 = mul_tensor_15 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_3, input_62);  add_tensor_3 = input_62 = None
        permute_default_3 = torch.ops.aten.permute.default(input_63, [1, 0]);  input_63 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_3, permute_default_3);  permute_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_64);  mm_default_2 = input_64 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_4, view_default_4, alpha = 1);  mul_tensor_4 = view_default_4 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(add_tensor_4, input_65);  input_65 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(mul_tensor_16, 768)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [2], True, dtype = None)
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_16, input_66);  mul_tensor_16 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [2], True, dtype = None);  mul_tensor_18 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(input_66, sum_dim_int_list_3);  sum_dim_int_list_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_17, sum_dim_int_list_2, alpha = 1);  mul_tensor_17 = sum_dim_int_list_2 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(add_tensor_4, input_66);  input_66 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1], False, dtype = None);  mul_tensor_20 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0, 1], False, dtype = None);  add_tensor_4 = None
        return (mm_default_1, mul_tensor_14, view_default_3, mul_tensor_19, sub_tensor_3, sum_dim_int_list_4, sum_dim_int_list_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
