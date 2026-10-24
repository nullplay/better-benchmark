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
    def forward(self, input_23, input_24, input_25, input_26, input_27, input_28, input_29):
        where_self = torch.ops.aten.where.self(input_23, input_24, input_25);  input_23 = input_24 = input_25 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(where_self, torch.bfloat16);  where_self = None
        mm_default = torch.ops.aten.mm.default(convert_element_type_default, input_26);  convert_element_type_default = input_26 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_27, torch.float32);  input_27 = None
        le_scalar = torch.ops.aten.le.Scalar(convert_element_type_default_2, 0)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.7580993408473766);  mul_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, exp_default);  mul_tensor_1 = exp_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.0507009873554805);  convert_element_type_default_1 = None
        where_self_1 = torch.ops.aten.where.self(le_scalar, mul_tensor_3, mul_tensor_4);  le_scalar = mul_tensor_3 = mul_tensor_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(where_self_1, torch.bfloat16);  where_self_1 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0])
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(input_28, [0, 1, 0, 0], 0);  input_28 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, constant_pad_nd_default);  permute_default = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default_1, 1, 0, -1, 1);  mm_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_3, [0], True, dtype = torch.float32);  convert_element_type_default_3 = None
        view_default = torch.ops.aten.view.default(sum_dim_int_list, input_29);  sum_dim_int_list = input_29 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        return (convert_element_type_default_5, convert_element_type_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
