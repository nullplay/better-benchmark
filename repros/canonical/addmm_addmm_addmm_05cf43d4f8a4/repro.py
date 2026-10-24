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
    def forward(self, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_49, torch.bfloat16);  input_49 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        add_tensor = torch.ops.aten.add.Tensor(input_51, input_52, alpha = 1);  input_51 = input_52 = None
        view_default = torch.ops.aten.view.default(input_53, input_54);  input_53 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_55, [1], False, dtype = None);  input_55 = None
        view_default_1 = torch.ops.aten.view.default(sum_dim_int_list, input_54);  sum_dim_int_list = input_54 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mean_dim, 1e-05, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default);  convert_element_type_default_2 = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_56, mul_tensor);  input_56 = mul_tensor = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_3, input_57)
        permute_default = torch.ops.aten.permute.default(input_58, [1, 0]);  input_58 = None
        addmm_default = torch.ops.aten.addmm.default(input_59, view_default_2, permute_default, beta = 1, alpha = 1);  input_59 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default, input_60);  addmm_default = input_60 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_61);  view_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        split_tensor = torch.ops.aten.split.Tensor(permute_default_1, 32, -1);  permute_default_1 = None
        getitem = split_tensor[0]
        getitem_1 = split_tensor[1];  split_tensor = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_3, input_57)
        permute_default_2 = torch.ops.aten.permute.default(input_62, [1, 0]);  input_62 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_63, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_63 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_64);  addmm_default_1 = input_64 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_61);  view_default_6 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        split_tensor_1 = torch.ops.aten.split.Tensor(permute_default_3, 32, -1);  permute_default_3 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(convert_element_type_default, 1);  convert_element_type_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(getitem, unsqueeze_default)
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_default_1)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_2, mul_tensor_3, alpha = 1);  mul_tensor_2 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(getitem_1, unsqueeze_default);  getitem_1 = unsqueeze_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(getitem, unsqueeze_default_1);  getitem = unsqueeze_default_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5, alpha = 1);  mul_tensor_4 = mul_tensor_5 = None
        cat_default = torch.ops.aten.cat.default([sub_tensor, add_tensor_4], -1);  sub_tensor = add_tensor_4 = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_3, input_57);  convert_element_type_default_3 = input_57 = None
        permute_default_4 = torch.ops.aten.permute.default(input_65, [1, 0]);  input_65 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_66, view_default_8, permute_default_4, beta = 1, alpha = 1);  input_66 = view_default_8 = permute_default_4 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default_2, input_67);  addmm_default_2 = input_67 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_61);  view_default_9 = input_61 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(permute_default_5, 2)
        slice_tensor = torch.ops.aten.slice.Tensor(permute_default_5, 2, -127, 9223372036854775807, 1);  permute_default_5 = None
        return (split_tensor_1, cat_default, unsqueeze_default_2, slice_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
