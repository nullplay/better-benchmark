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
    def forward(self, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39):
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(input_27, [0, -4], 0);  input_27 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(input_28, [0, -4], 0);  input_28 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(input_29, [0, -4], 0);  input_29 = None
        cat_default = torch.ops.aten.cat.default([constant_pad_nd_default_2, constant_pad_nd_default_1, constant_pad_nd_default], 0);  constant_pad_nd_default_2 = constant_pad_nd_default_1 = constant_pad_nd_default = None
        view_default = torch.ops.aten.view.default(cat_default, input_30);  cat_default = input_30 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 3, 0, 2, 4]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_31);  clone_default = input_31 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_32);  view_default_1 = input_32 = None
        mm_default = torch.ops.aten.mm.default(view_default_2, input_33);  input_33 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_1, input_34);  permute_default_1 = input_34 = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_35);  mm_default = input_35 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_36);  input_36 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 240)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(input_37, input_38, alpha = 1);  input_37 = input_38 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_39);  sub_tensor = input_39 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], False, dtype = None);  mul_tensor_5 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [0, 1], False, dtype = None);  convert_element_type_default = None
        return (convert_element_type_default_1, mul_tensor_4, sub_tensor_1, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
