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
    def forward(self, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71):
        full_default = torch.ops.aten.full.default(input_54, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        select_scatter_default = torch.ops.aten.select_scatter.default(full_default, input_55, 1, 0);  full_default = input_55 = None
        view_default = torch.ops.aten.view.default(select_scatter_default, input_56)
        mm_default = torch.ops.aten.mm.default(view_default, input_57);  input_57 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_58);  permute_default = input_58 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_59);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_60, input_59);  input_60 = input_59 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476)
        erf_default = torch.ops.aten.erf.default(mul_tensor);  mul_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, 0.5);  add_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default_2)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, -0.5);  mul_tensor_2 = None
        exp_default = torch.ops.aten.exp.default(mul_tensor_3);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(exp_default, 0.3989422804014327);  exp_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_4);  convert_element_type_default_2 = mul_tensor_4 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_5, alpha = 1);  mul_tensor_1 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, add_tensor_1);  convert_element_type_default_1 = add_tensor_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_3, input_61);  convert_element_type_default_3 = input_61 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_3, input_62);  input_62 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_63);  permute_default_1 = input_63 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_3, [0], True, dtype = torch.float32);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_64);  sum_dim_int_list = input_64 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_4, torch.bfloat16);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_54);  mm_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_65, 768)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_65, [2], True, dtype = None);  input_65 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_66, input_67);  input_66 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(input_68, [2], True, dtype = None);  input_68 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, sum_dim_int_list_2);  mul_tensor_8 = sum_dim_int_list_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_7, sum_dim_int_list_1, alpha = 1);  mul_tensor_7 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_9, alpha = 1);  sub_tensor = mul_tensor_9 = None
        div_tensor = torch.ops.aten.div.Tensor(input_67, 768);  input_67 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_1);  div_tensor = sub_tensor_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(select_scatter_default, convert_element_type_default_7, alpha = 1);  select_scatter_default = convert_element_type_default_7 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_2, input_56);  add_tensor_2 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_6, input_69);  input_69 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [1, 0]);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(input_70, input_54);  input_70 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_56);  view_default_7 = input_56 = None
        mm_default_5 = torch.ops.aten.mm.default(permute_default_2, view_default_8);  permute_default_2 = view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_4, input_54);  mm_default_4 = input_54 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_71);  view_default_9 = input_71 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        return (convert_element_type_default, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_8, permute_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
