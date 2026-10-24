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
    def forward(self, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36):
        mm_default = torch.ops.aten.mm.default(input_25, input_26);  input_26 = None
        permute_default = torch.ops.aten.permute.default(input_25, [1, 0]);  input_25 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_27);  permute_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        le_scalar = torch.ops.aten.le.Scalar(input_27, 0);  input_27 = None
        full_default = torch.ops.aten.full.default(input_28, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_28 = None
        where_self = torch.ops.aten.where.self(le_scalar, full_default, mm_default);  le_scalar = mm_default = None
        mm_default_2 = torch.ops.aten.mm.default(where_self, input_29);  input_29 = None
        permute_default_1 = torch.ops.aten.permute.default(where_self, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_30);  permute_default_1 = input_30 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(where_self, [0], True, dtype = torch.float32);  where_self = None
        view_default = torch.ops.aten.view.default(sum_dim_int_list, input_31);  sum_dim_int_list = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32);  convert_element_type_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_32, torch.bfloat16);  input_32 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 2.0);  convert_element_type_default_4 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mm_default_2, mul_tensor);  mm_default_2 = mul_tensor = None
        where_self_1 = torch.ops.aten.where.self(input_33, full_default, mul_tensor_1);  input_33 = full_default = mul_tensor_1 = None
        mm_default_4 = torch.ops.aten.mm.default(where_self_1, input_34);  input_34 = None
        permute_default_2 = torch.ops.aten.permute.default(where_self_1, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_2, input_35);  permute_default_2 = input_35 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(where_self_1, [0], True, dtype = torch.float32);  where_self_1 = None
        view_default_1 = torch.ops.aten.view.default(sum_dim_int_list_1, input_31);  sum_dim_int_list_1 = input_31 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mm_default_4, input_36);  mm_default_4 = input_36 = None
        return (convert_element_type_default, convert_element_type_default_2, convert_element_type_default_3, mm_default_5, view_default_1, mul_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
