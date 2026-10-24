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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62):
        sub_tensor = torch.ops.aten.sub.Tensor(input_44, input_45, alpha = 1);  input_44 = input_45 = None
        div_tensor = torch.ops.aten.div.Tensor(input_46, 4096);  input_46 = None
        mul_tensor = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor);  div_tensor = sub_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_47)
        mm_default = torch.ops.aten.mm.default(view_default, input_48);  view_default = input_48 = None
        view_default_1 = torch.ops.aten.view.default(input_49, input_47);  input_49 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, input_50);  view_default_1 = input_50 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_51);  mm_default_1 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, view_default_2, alpha = 1);  convert_element_type_default = view_default_2 = None
        clone_default = torch.ops.aten.clone.default(input_52, memory_format = torch.contiguous_format);  input_52 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_47);  clone_default = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_3, input_53);  view_default_3 = input_53 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_51);  mm_default_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_4, alpha = 1);  add_tensor = view_default_4 = None
        clone_default_1 = torch.ops.aten.clone.default(input_54, memory_format = torch.contiguous_format);  input_54 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_51);  clone_default_1 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_47);  view_default_5 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_6, input_55);  input_55 = None
        permute_default = torch.ops.aten.permute.default(view_default_6, [1, 0]);  view_default_6 = None
        mm_default_4 = torch.ops.aten.mm.default(permute_default, input_56);  permute_default = input_56 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_3, input_51);  mm_default_3 = input_51 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_7, alpha = 1);  add_tensor_1 = view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_2, input_47);  add_tensor_2 = input_47 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_8, input_57);  input_57 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        mm_default_6 = torch.ops.aten.mm.default(permute_default_1, input_58);  permute_default_1 = input_58 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_8, [0], True, dtype = torch.float32);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(sum_dim_int_list, input_59);  sum_dim_int_list = input_59 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_9, torch.bfloat16);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_5, input_60);  mm_default_5 = input_60 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(convert_element_type_default_1, torch.float32);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_61);  input_61 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 128)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True, dtype = None)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_62);  mul_tensor_1 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_62, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list_1, alpha = 1);  mul_tensor_2 = sum_dim_int_list_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_62);  input_62 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], False, dtype = None);  mul_tensor_5 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 1], False, dtype = None);  convert_element_type_default_2 = None
        return (mm_default, mm_default_4, convert_element_type_default_3, convert_element_type_default_4, mul_tensor_4, sub_tensor_1, sum_dim_int_list_3, sum_dim_int_list_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
