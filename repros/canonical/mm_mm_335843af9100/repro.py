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
    def forward(self, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26):
        mul_tensor = torch.ops.aten.mul.Tensor(input_18, input_19);  input_18 = input_19 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_20, torch.float32);  input_20 = None
        exp_default = torch.ops.aten.exp.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [1], True, dtype = None)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_tensor_1, alpha = 1);  convert_element_type_default_1 = mul_tensor_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_3, input_21);  convert_element_type_default_3 = input_21 = None
        add_tensor = torch.ops.aten.add.Tensor(input_22, view_default, alpha = 1);  input_22 = view_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_23);  add_tensor = input_23 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_24);  permute_default = input_24 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, input_25);  view_default_1 = input_25 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_26);  mm_default_1 = input_26 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        return (convert_element_type_default_4, convert_element_type_default_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
