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
    def forward(self, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70):
        full_default = torch.ops.aten.full.default(input_48, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_48 = None
        view_default = torch.ops.aten.view.default(input_49, input_50);  input_49 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_51);  convert_element_type_default = input_51 = None
        mm_default = torch.ops.aten.mm.default(input_52, input_53);  input_52 = input_53 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_50);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default_1, alpha = 1);  mul_tensor = convert_element_type_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], True, dtype = torch.float32)
        view_default_2 = torch.ops.aten.view.default(sum_dim_int_list, input_54);  sum_dim_int_list = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, input_55);  input_55 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor, input_56);  add_tensor = input_56 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True, dtype = torch.float32);  mul_tensor_1 = None
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list_1, input_54);  sum_dim_int_list_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16)
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_2, input_57);  convert_element_type_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_4, input_58);  input_58 = None
        permute_default = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_59);  permute_default = input_59 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True, dtype = torch.float32);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(sum_dim_int_list_2, input_54);  sum_dim_int_list_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_5, torch.bfloat16);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_1, input_60);  mm_default_1 = input_60 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float32);  convert_element_type_default_3 = None
        where_self = torch.ops.aten.where.self(input_61, full_default, view_default_6);  input_61 = full_default = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(where_self, input_62);  where_self = input_62 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_7, input_63);  input_63 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_1, input_64);  permute_default_1 = input_64 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_7, [0], True, dtype = torch.float32);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(sum_dim_int_list_3, input_65);  sum_dim_int_list_3 = input_65 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_3, input_50);  mm_default_3 = input_50 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_2, convert_element_type_default_6, alpha = 1);  mul_tensor_2 = convert_element_type_default_6 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], True, dtype = torch.float32)
        view_default_10 = torch.ops.aten.view.default(sum_dim_int_list_4, input_54);  sum_dim_int_list_4 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_1, input_66);  input_66 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_1, input_67);  add_tensor_1 = input_67 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], True, dtype = torch.float32);  mul_tensor_3 = None
        view_default_11 = torch.ops.aten.view.default(sum_dim_int_list_5, input_54);  sum_dim_int_list_5 = input_54 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_7, input_57);  convert_element_type_default_7 = input_57 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_68, torch.float32);  input_68 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_4, convert_element_type_default_8, alpha = 1);  mul_tensor_4 = convert_element_type_default_8 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True, dtype = torch.float32)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_2, input_69);  input_69 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_2, input_70);  add_tensor_2 = input_70 = None
        return (view_default_2, view_default_3, convert_element_type_default_4, convert_element_type_default_5, mm_default_4, view_default_8, view_default_10, view_default_11, view_default_12, sum_dim_int_list_6, mul_tensor_5, mul_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
