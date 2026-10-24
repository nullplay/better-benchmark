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
    def forward(self, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34):
        view_default = torch.ops.aten.view.default(input_25, input_26);  input_25 = input_26 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0])
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 7], 0);  permute_default = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, input_27);  constant_pad_nd_default = input_27 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -7, 1);  mm_default = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(view_default, [0, 7, 0, 0], 0);  view_default = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(input_28, [0, 0, 0, 7], 0);  input_28 = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_29);  mm_default_1 = input_29 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_30, convert_element_type_default, alpha = 1);  input_30 = convert_element_type_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, input_31);  input_31 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_32);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_32, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_33, sub_tensor_1);  input_33 = sub_tensor_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor, input_32);  input_32 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], False, dtype = None);  mul_tensor_5 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], False, dtype = None);  add_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default_1, input_34, alpha = 1);  convert_element_type_default_1 = input_34 = None
        return (mul_tensor_4, sum_dim_int_list_2, sum_dim_int_list_3, add_tensor_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
