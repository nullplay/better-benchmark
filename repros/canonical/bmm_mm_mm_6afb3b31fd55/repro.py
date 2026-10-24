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
    def forward(self, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56):
        bmm_default = torch.ops.aten.bmm.default(input_42, input_43);  input_42 = input_43 = None
        view_default = torch.ops.aten.view.default(bmm_default, input_44);  bmm_default = input_44 = None
        mul_scalar = torch.ops.aten.mul.Scalar(view_default, 0.4204482076268573);  view_default = None
        permute_default = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(permute_default_1, input_45);  permute_default_1 = None
        clone_default = torch.ops.aten.clone.default(view_default_1, memory_format = torch.contiguous_format);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_46);  clone_default = None
        mm_default = torch.ops.aten.mm.default(view_default_2, input_47);  input_47 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_48);  permute_default_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True, dtype = torch.float32);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list, input_49);  sum_dim_int_list = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_3, torch.bfloat16);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mm_default, input_45);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        view_default_5 = torch.ops.aten.view.default(input_50, input_45);  input_50 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_46);  view_default_5 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_6, input_51);  input_51 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_6, [1, 0]);  view_default_6 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_3, input_48);  permute_default_3 = input_48 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_2, input_45);  mm_default_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default_1, convert_element_type_default_4, alpha = 1);  convert_element_type_default_1 = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(add_tensor, [0, 1], True, dtype = torch.float32)
        view_default_8 = torch.ops.aten.view.default(sum_dim_int_list_1, input_49);  sum_dim_int_list_1 = None
        view_default_9 = torch.ops.aten.view.default(input_52, input_45);  input_52 = input_45 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, view_default_9);  view_default_9 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, input_53);  add_tensor = input_53 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1], True, dtype = torch.float32);  mul_tensor = None
        view_default_10 = torch.ops.aten.view.default(sum_dim_int_list_2, input_49);  sum_dim_int_list_2 = None
        view_default_11 = torch.ops.aten.view.default(convert_element_type_default_6, input_46);  convert_element_type_default_6 = input_46 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_11, input_54);  input_54 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_11, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_4, input_55);  permute_default_4 = input_55 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_11, [0], True, dtype = torch.float32);  view_default_11 = None
        view_default_12 = torch.ops.aten.view.default(sum_dim_int_list_3, input_49);  sum_dim_int_list_3 = input_49 = None
        view_default_13 = torch.ops.aten.view.default(mm_default_4, input_56);  mm_default_4 = input_56 = None
        return (convert_element_type_default_2, convert_element_type_default_3, convert_element_type_default_5, view_default_8, view_default_10, mm_default_5, view_default_12, view_default_13)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
