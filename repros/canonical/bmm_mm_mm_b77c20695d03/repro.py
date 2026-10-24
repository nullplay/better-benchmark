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
    def forward(self, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74):
        mul_tensor = torch.ops.aten.mul.Tensor(input_56, 1536)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_56, [2], True, dtype = None);  input_56 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_57, [2], True, dtype = None);  input_57 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_58, sum_dim_int_list_1);  input_58 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, sum_dim_int_list, alpha = 1);  mul_tensor = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_1, alpha = 1);  sub_tensor = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_59, sub_tensor_1);  input_59 = sub_tensor_1 = None
        bmm_default = torch.ops.aten.bmm.default(input_60, input_61);  input_60 = input_61 = None
        full_default = torch.ops.aten.full.default(input_62, 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False);  input_62 = None
        div_tensor = torch.ops.aten.div.Tensor(bmm_default, full_default);  bmm_default = full_default = None
        permute_default = torch.ops.aten.permute.default(div_tensor, [0, 2, 1]);  div_tensor = None
        view_default = torch.ops.aten.view.default(input_63, input_64);  input_63 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_65);  view_default = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_66);  input_66 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_1, input_67);  permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_64);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_2, convert_element_type_default, alpha = 1);  mul_tensor_2 = convert_element_type_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_3 = torch.ops.aten.view.default(permute_default, input_68);  permute_default = input_68 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_2, input_64);  permute_default_2 = None
        clone_default = torch.ops.aten.clone.default(view_default_4, memory_format = torch.contiguous_format);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_65);  clone_default = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_5, input_69);  input_69 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_3, input_67);  permute_default_3 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True, dtype = torch.float32);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list_2, input_70);  sum_dim_int_list_2 = input_70 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_6, torch.bfloat16);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_2, input_64);  mm_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_3, alpha = 1);  add_tensor = convert_element_type_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        view_default_8 = torch.ops.aten.view.default(input_71, input_64);  input_71 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_65);  view_default_8 = input_65 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_9, input_72);  input_72 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_9, [1, 0]);  view_default_9 = None
        mm_default_5 = torch.ops.aten.mm.default(permute_default_4, input_67);  permute_default_4 = input_67 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_4, input_64);  mm_default_4 = input_64 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_6, alpha = 1);  add_tensor_1 = convert_element_type_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_2, input_73);  input_73 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 1536)
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_74);  mul_tensor_3 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None);  mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_74, sum_dim_int_list_4);  sum_dim_int_list_4 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list_3, alpha = 1);  mul_tensor_4 = sum_dim_int_list_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_6, alpha = 1);  sub_tensor_2 = mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(add_tensor_2, input_74);  input_74 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], False, dtype = None);  mul_tensor_7 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], False, dtype = None);  add_tensor_2 = None
        return (convert_element_type_default_1, convert_element_type_default_4, convert_element_type_default_5, convert_element_type_default_7, sub_tensor_3, sum_dim_int_list_5, sum_dim_int_list_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
