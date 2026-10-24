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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49):
        full_default = torch.ops.aten.full.default(input_35, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default = torch.ops.aten.select_scatter.default(full_default, input_36, 1, 0);  full_default = input_36 = None
        view_default = torch.ops.aten.view.default(select_scatter_default, input_37)
        mm_default = torch.ops.aten.mm.default(view_default, input_38);  input_38 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_39);  permute_default = input_39 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_40);  mm_default = input_40 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, input_41);  convert_element_type_default_1 = input_41 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_42, input_43);  input_42 = input_43 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(input_44, input_45, alpha = 1);  input_44 = input_45 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor, input_46);  sub_tensor = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True, dtype = None);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list, alpha = 1);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5, alpha = 1);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor = torch.ops.aten.div.Tensor(input_46, 768);  input_46 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        add_tensor = torch.ops.aten.add.Tensor(select_scatter_default, convert_element_type_default_2, alpha = 1);  select_scatter_default = convert_element_type_default_2 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor, input_37);  add_tensor = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_47);  input_47 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(input_48, [0, 2, 1, 3]);  input_48 = None
        view_default_3 = torch.ops.aten.view.default(permute_default_2, input_35);  permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_37);  view_default_3 = input_37 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, view_default_4);  permute_default_1 = view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_35);  mm_default_2 = input_35 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_49);  view_default_5 = input_49 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        return (convert_element_type_default, mul_tensor, convert_element_type_default_3, permute_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
