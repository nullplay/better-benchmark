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
    def forward(self, input_17, input_18, input_19, input_20, input_21, input_22, input_23):
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_17, [-1], True, dtype = None)
        exp_default = torch.ops.aten.exp.default(input_18);  input_18 = None
        mul_tensor = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_17, mul_tensor, alpha = 1);  input_17 = mul_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_19);  convert_element_type_default = input_19 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default, [0, 3, 0, 0], 0)
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(input_20, [0, 0, 0, 3], 0);  input_20 = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 3], 0);  permute_default = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default_2, input_21);  constant_pad_nd_default_2 = input_21 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -3, 1);  mm_default_1 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_22);  mm_default = input_22 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default_1, input_23, alpha = 1);  convert_element_type_default_1 = input_23 = None
        return (convert_element_type_default_2, add_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
