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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57):
        bmm_default = torch.ops.aten.bmm.default(input_44, input_45);  input_44 = input_45 = None
        permute_default = torch.ops.aten.permute.default(bmm_default, [0, 2, 1]);  bmm_default = None
        view_default = torch.ops.aten.view.default(input_46, input_47);  input_46 = None
        view_default_1 = torch.ops.aten.view.default(permute_default, input_47);  permute_default = input_47 = None
        permute_default_1 = torch.ops.aten.permute.default(input_48, [0, 2, 1, 3]);  input_48 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_49);  clone_default = None
        permute_default_2 = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(clone_default_1, input_49);  clone_default_1 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_3, input_49);  permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(view_default_3, input_50);  view_default_3 = None
        mm_default = torch.ops.aten.mm.default(view_default_5, input_51);  input_51 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_5, [1, 0]);  view_default_5 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_4, input_52);  permute_default_4 = None
        view_default_6 = torch.ops.aten.view.default(mm_default, input_49);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        clone_default_2 = torch.ops.aten.clone.default(view_default_4, memory_format = torch.contiguous_format);  view_default_4 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_2, input_50);  clone_default_2 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_7, input_53);  input_53 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_5, input_52);  permute_default_5 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_7, [0], True, dtype = torch.float32);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(sum_dim_int_list, input_54);  sum_dim_int_list = input_54 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_8, torch.bfloat16);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_2, input_49);  mm_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_3, alpha = 1);  convert_element_type_default = convert_element_type_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_2, 0.125);  view_default_2 = None
        view_default_10 = torch.ops.aten.view.default(mul_tensor, input_50);  mul_tensor = input_50 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_10, input_55);  input_55 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_10, [1, 0]);  view_default_10 = None
        mm_default_5 = torch.ops.aten.mm.default(permute_default_6, input_52);  permute_default_6 = input_52 = None
        view_default_11 = torch.ops.aten.view.default(mm_default_4, input_49);  mm_default_4 = input_49 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32);  view_default_11 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_6, alpha = 1);  add_tensor = convert_element_type_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_1, input_56);  input_56 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_1, input_57);  input_57 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], False, dtype = None);  add_tensor_1 = None
        return (convert_element_type_default_1, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, mul_tensor_1, mul_tensor_2, sum_dim_int_list_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
