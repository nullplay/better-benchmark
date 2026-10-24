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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44):
        mul_tensor = torch.ops.aten.mul.Tensor(input_30, input_31);  input_30 = input_31 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_32);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_32, sum_dim_int_list_1);  input_32 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_33, sub_tensor_1);  input_33 = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_34);  convert_element_type_default = input_34 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_35);  input_35 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_36);  permute_default = input_36 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_37);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_38);  view_default_1 = input_38 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_39);  clone_default = input_39 = None
        mm_default_2 = torch.ops.aten.mm.default(input_40, input_41);  input_40 = input_41 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_37);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_4, convert_element_type_default_2, alpha = 1);  mul_tensor_4 = convert_element_type_default_2 = None
        view_default_5 = torch.ops.aten.view.default(input_42, input_37);  input_42 = input_37 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_3, alpha = 1);  add_tensor = convert_element_type_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_43, torch.float32);  input_43 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_4, alpha = 1);  add_tensor_1 = convert_element_type_default_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_2, input_44);  add_tensor_2 = input_44 = None
        return (convert_element_type_default_1, view_default_3, mul_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
