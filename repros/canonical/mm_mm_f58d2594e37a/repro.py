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
    def forward(self, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28):
        mm_default = torch.ops.aten.mm.default(input_21, input_22);  input_22 = None
        permute_default = torch.ops.aten.permute.default(input_21, [1, 0]);  input_21 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_23);  permute_default = input_23 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        full_default = torch.ops.aten.full.default(input_24, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_24 = None
        select_scatter_default = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default, 1, 0);  full_default = convert_element_type_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(select_scatter_default, input_25);  input_25 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 192)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_26);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_26, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_27, sub_tensor_1);  input_27 = sub_tensor_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(select_scatter_default, input_26);  input_26 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], False, dtype = None);  mul_tensor_5 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(select_scatter_default, [0, 1], False, dtype = None);  select_scatter_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_4, input_28, alpha = 1);  mul_tensor_4 = input_28 = None
        return (convert_element_type_default_1, sum_dim_int_list_2, sum_dim_int_list_3, convert_element_type_default_2, add_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
