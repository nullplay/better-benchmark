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
    def forward(self, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57):
        mm_default = torch.ops.aten.mm.default(input_40, input_41);  input_41 = None
        permute_default = torch.ops.aten.permute.default(input_40, [1, 0]);  input_40 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_42);  permute_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        le_scalar = torch.ops.aten.le.Scalar(input_42, 0);  input_42 = None
        full_default = torch.ops.aten.full.default(input_43, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_43 = None
        where_self = torch.ops.aten.where.self(le_scalar, full_default, mm_default);  le_scalar = mm_default = None
        mm_default_2 = torch.ops.aten.mm.default(where_self, input_44);  input_44 = None
        permute_default_1 = torch.ops.aten.permute.default(where_self, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_45);  permute_default_1 = input_45 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(where_self, [0], True, dtype = torch.float32);  where_self = None
        view_default = torch.ops.aten.view.default(sum_dim_int_list, input_46);  sum_dim_int_list = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32);  convert_element_type_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_47, torch.bfloat16);  input_47 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 2.0);  convert_element_type_default_4 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mm_default_2, mul_tensor);  mm_default_2 = mul_tensor = None
        where_self_1 = torch.ops.aten.where.self(input_48, full_default, mul_tensor_1);  input_48 = full_default = mul_tensor_1 = None
        mm_default_4 = torch.ops.aten.mm.default(where_self_1, input_49);  input_49 = None
        permute_default_2 = torch.ops.aten.permute.default(where_self_1, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_2, input_50);  permute_default_2 = input_50 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(where_self_1, [0], True, dtype = torch.float32);  where_self_1 = None
        view_default_1 = torch.ops.aten.view.default(sum_dim_int_list_1, input_46);  sum_dim_int_list_1 = input_46 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32);  convert_element_type_default_5 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, 2.0);  convert_element_type_default_8 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mm_default_4, mul_tensor_2);  mm_default_4 = mul_tensor_2 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_3, input_52);  mul_tensor_3 = input_52 = None
        _adaptive_avg_pool2d_backward_default = torch.ops.aten._adaptive_avg_pool2d_backward.default(view_default_2, input_53);  view_default_2 = input_53 = None
        full_default_1 = torch.ops.aten.full.default(input_54, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_54 = None
        view_default_3 = torch.ops.aten.view.default(_adaptive_avg_pool2d_backward_default, input_55);  _adaptive_avg_pool2d_backward_default = None
        _low_memory_max_pool_offsets_to_indices_default = torch.ops.prims._low_memory_max_pool_offsets_to_indices.default(input_56, [3, 3], [13, 13], [2, 2], [0, 0], [1, 1]);  input_56 = None
        view_default_4 = torch.ops.aten.view.default(_low_memory_max_pool_offsets_to_indices_default, input_55);  _low_memory_max_pool_offsets_to_indices_default = input_55 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        scatter_add_default = torch.ops.aten.scatter_add.default(full_default_1, 1, view_default_4, convert_element_type_default_9);  full_default_1 = view_default_4 = convert_element_type_default_9 = None
        view_default_5 = torch.ops.aten.view.default(scatter_add_default, input_57);  scatter_add_default = input_57 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_5, torch.bfloat16);  view_default_5 = None
        return (convert_element_type_default, convert_element_type_default_2, convert_element_type_default_3, convert_element_type_default_6, convert_element_type_default_7, convert_element_type_default_10)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
