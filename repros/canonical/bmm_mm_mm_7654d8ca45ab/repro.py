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
    def forward(self, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51):
        bmm_default = torch.ops.aten.bmm.default(input_34, input_35);  input_34 = input_35 = None
        view_default = torch.ops.aten.view.default(bmm_default, input_36);  bmm_default = None
        view_default_1 = torch.ops.aten.view.default(input_37, input_38);  input_37 = input_38 = None
        view_default_2 = torch.ops.aten.view.default(input_39, input_36);  input_39 = input_36 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 2]);  view_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_2, 0.1767766952966369);  view_default_2 = None
        cat_default = torch.ops.aten.cat.default([mul_tensor, permute_default, view_default], 0);  mul_tensor = permute_default = view_default = None
        view_default_3 = torch.ops.aten.view.default(cat_default, input_40);  cat_default = input_40 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 3, 0, 2, 4]);  view_default_3 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_41);  clone_default = input_41 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_42);  view_default_4 = input_42 = None
        mm_default = torch.ops.aten.mm.default(view_default_5, input_43);  input_43 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_44);  permute_default_2 = input_44 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True, dtype = torch.float32);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list, input_45);  sum_dim_int_list = input_45 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_6, torch.bfloat16);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(mm_default, input_46);  mm_default = input_46 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_1, input_47);  convert_element_type_default_1 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_48);  view_default_8 = input_48 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_9, [0, 1, 3, 2, 4, 5]);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(permute_default_3, input_47);  permute_default_3 = input_47 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default_10, input_49);  input_49 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 1024)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [3], True, dtype = None)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_50, input_51);  input_50 = input_51 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_10, mul_tensor_3);  mul_tensor_3 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2], False, dtype = None);  mul_tensor_5 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_10, [0, 1, 2], False, dtype = None);  view_default_10 = None
        return (convert_element_type_default_2, convert_element_type_default_3, mul_tensor_2, sum_dim_int_list_1, mul_tensor_4, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
