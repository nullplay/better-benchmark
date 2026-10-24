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
    def forward(self, input_17, input_18, input_19, input_20, input_21, input_22):
        mm_default = torch.ops.aten.mm.default(input_17, input_18);  input_18 = None
        permute_default = torch.ops.aten.permute.default(input_17, [1, 0]);  input_17 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_19);  permute_default = input_19 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        full_default = torch.ops.aten.full.default(input_20, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_20 = None
        select_scatter_default = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default, 1, 0);  full_default = convert_element_type_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(select_scatter_default, input_21);  input_21 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 192)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_22);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_22, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(select_scatter_default, input_22);  input_22 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], False, dtype = None);  mul_tensor_4 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(select_scatter_default, [0, 1], False, dtype = None);  select_scatter_default = None
        return (convert_element_type_default_1, mul_tensor_3, sub_tensor, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
