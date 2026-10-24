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
    def forward(self, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83):
        view_default = torch.ops.aten.view.default(input_63, input_64);  input_63 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default, input_65);  view_default = input_65 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_66);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_66, sum_dim_int_list_1);  input_66 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_67, sub_tensor_1);  input_67 = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_68, torch.float32);  input_68 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_5);  mul_tensor_5 = None
        view_default_1 = torch.ops.aten.view.default(mul_tensor_6, input_69);  mul_tensor_6 = input_69 = None
        permute_default = torch.ops.aten.permute.default(input_70, [1, 0]);  input_70 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default_1);  permute_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_71);  permute_default_2 = input_71 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_72);  mm_default = None
        view_default_3 = torch.ops.aten.view.default(input_73, input_72);  input_73 = input_72 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_3, 0.5)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_default_2, mul_tensor_7);  mul_tensor_7 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_3, input_74, alpha = 1);  input_74 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_9);  mul_tensor_9 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(view_default_2, add_tensor_1);  view_default_2 = add_tensor_1 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(tanh_default, tanh_default);  tanh_default = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, mul_tensor_11, alpha = 1);  mul_tensor_11 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_8, sub_tensor_2);  mul_tensor_8 = sub_tensor_2 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_12, 0.7978845608028654);  mul_tensor_12 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_13, 0.044715)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(view_default_3, 2.0);  view_default_3 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 3.0);  pow_tensor_scalar = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_14, mul_scalar);  mul_tensor_14 = mul_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_13, mul_tensor_15, alpha = 1);  mul_tensor_13 = mul_tensor_15 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_10, 0.5);  mul_tensor_10 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_16, alpha = 1);  add_tensor_2 = mul_tensor_16 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_3, input_75);  add_tensor_3 = input_75 = None
        permute_default_3 = torch.ops.aten.permute.default(input_76, [1, 0]);  input_76 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_3, [1, 0]);  permute_default_3 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, permute_default_4);  permute_default_4 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_5, input_77);  permute_default_5 = input_77 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True, dtype = None);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(sum_dim_int_list_2, input_78);  sum_dim_int_list_2 = input_78 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_64);  mm_default_2 = input_64 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_4, view_default_6, alpha = 1);  mul_tensor_4 = view_default_6 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(add_tensor_4, input_79);  input_79 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(mul_tensor_17, 768)
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [2], True, dtype = None)
        mul_tensor_19 = torch.ops.aten.mul.Tensor(mul_tensor_17, input_80);  mul_tensor_17 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [2], True, dtype = None);  mul_tensor_19 = None
        mul_tensor_20 = torch.ops.aten.mul.Tensor(input_80, sum_dim_int_list_4);  sum_dim_int_list_4 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_18, sum_dim_int_list_3, alpha = 1);  mul_tensor_18 = sum_dim_int_list_3 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_20, alpha = 1);  sub_tensor_3 = mul_tensor_20 = None
        mul_tensor_21 = torch.ops.aten.mul.Tensor(input_81, sub_tensor_4);  input_81 = sub_tensor_4 = None
        mul_tensor_22 = torch.ops.aten.mul.Tensor(add_tensor_4, input_80);  input_80 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1], False, dtype = None);  mul_tensor_22 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(add_tensor_4, [0, 1], False, dtype = None);  add_tensor_4 = None
        select_scatter_default = torch.ops.aten.select_scatter.default(input_82, mul_tensor_21, 3, 0);  input_82 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_21, input_83, alpha = 1);  mul_tensor_21 = input_83 = None
        return (mm_default_1, mm_default_3, view_default_5, sum_dim_int_list_5, sum_dim_int_list_6, select_scatter_default, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
