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
    def forward(self, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_20, torch.bfloat16);  input_20 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        exp_default = torch.ops.aten.exp.default(input_21);  input_21 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [1], True, dtype = None)
        mul_tensor = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_tensor, alpha = 1);  convert_element_type_default_1 = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_22);  convert_element_type_default_2 = input_22 = None
        add_tensor = torch.ops.aten.add.Tensor(input_23, view_default, alpha = 1);  input_23 = view_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_24);  add_tensor = input_24 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0])
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 7], 0);  permute_default = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, input_25);  constant_pad_nd_default = input_25 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 0, 0, -7, 1);  mm_default = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(view_default_1, [0, 7, 0, 0], 0);  view_default_1 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(input_26, [0, 0, 0, 7], 0);  input_26 = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default_1, constant_pad_nd_default_2);  constant_pad_nd_default_1 = constant_pad_nd_default_2 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_27);  mm_default_1 = input_27 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        return (convert_element_type_default_3, convert_element_type_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
