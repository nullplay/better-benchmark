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
    def forward(self, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48):
        mm_default = torch.ops.aten.mm.default(input_32, input_33);  input_33 = None
        permute_default = torch.ops.aten.permute.default(input_32, [1, 0]);  input_32 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_34);  permute_default = input_34 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_35);  convert_element_type_default = None
        expand_default = torch.ops.aten.expand.default(view_default, input_35, implicit = False);  view_default = input_35 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 1);  expand_default = None
        full_default = torch.ops.aten.full.default(input_36, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_36 = None
        where_self = torch.ops.aten.where.self(input_37, full_default, div_scalar);  input_37 = div_scalar = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_38, torch.float32);  input_38 = None
        mul_tensor = torch.ops.aten.mul.Tensor(where_self, convert_element_type_default_2);  convert_element_type_default_2 = None
        view_default_1 = torch.ops.aten.view.default(mul_tensor, input_39);  mul_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_1, [2], False, dtype = None);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(where_self, input_39)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_2, [2], False, dtype = None);  view_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, input_40);  input_40 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_41, -1);  input_41 = None
        view_default_3 = torch.ops.aten.view.default(where_self, input_42)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_3, unsqueeze_default);  view_default_3 = unsqueeze_default = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_2, input_43, alpha = 1);  mul_tensor_2 = input_43 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_1, input_44);  input_44 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(sum_dim_int_list_1, [0], False, dtype = None);  sum_dim_int_list_1 = None
        getitem = input_45[0];  input_45 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(getitem, torch.float32);  getitem = None
        add_tensor_1 = torch.ops.aten.add.Tensor(where_self, convert_element_type_default_3, alpha = 1);  where_self = convert_element_type_default_3 = None
        le_scalar = torch.ops.aten.le.Scalar(input_46, 0);  input_46 = None
        where_self_1 = torch.ops.aten.where.self(le_scalar, full_default, add_tensor_1);  le_scalar = full_default = add_tensor_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(where_self_1, input_47);  input_47 = None
        view_default_5 = torch.ops.aten.view.default(where_self_1, input_39);  input_39 = None
        view_default_6 = torch.ops.aten.view.default(where_self_1, input_42);  input_42 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(where_self_1, input_48);  where_self_1 = input_48 = None
        return (convert_element_type_default_1, sum_dim_int_list, mul_tensor_1, add_tensor, view_default_4, sum_dim_int_list_2, mul_tensor_3, view_default_5, view_default_6, mul_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
