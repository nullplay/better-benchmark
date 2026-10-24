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
    def forward(self, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29):
        mm_default = torch.ops.aten.mm.default(input_19, input_20);  input_20 = None
        permute_default = torch.ops.aten.permute.default(input_19, [1, 0]);  input_19 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_21);  permute_default = input_21 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_22, torch.bfloat16);  input_22 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 2.0);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mm_default, mul_tensor);  mm_default = mul_tensor = None
        full_default = torch.ops.aten.full.default(input_23, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_23 = None
        where_self = torch.ops.aten.where.self(input_24, full_default, mul_tensor_1);  input_24 = mul_tensor_1 = None
        mm_default_2 = torch.ops.aten.mm.default(where_self, input_25);  input_25 = None
        permute_default_1 = torch.ops.aten.permute.default(where_self, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_26);  permute_default_1 = input_26 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(where_self, [0], True, dtype = torch.float32);  where_self = None
        view_default = torch.ops.aten.view.default(sum_dim_int_list, input_27);  sum_dim_int_list = input_27 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_28, 2.0);  input_28 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mm_default_2, mul_tensor_2);  mm_default_2 = mul_tensor_2 = None
        where_self_1 = torch.ops.aten.where.self(input_29, full_default, mul_tensor_3);  input_29 = full_default = mul_tensor_3 = None
        return (convert_element_type_default, convert_element_type_default_2, convert_element_type_default_3, where_self_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
