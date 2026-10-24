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
    def forward(self, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94):
        full_default = torch.ops.aten.full.default(input_70, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_70 = None
        full_default_1 = torch.ops.aten.full.default(input_71, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_71 = None
        add_tensor = torch.ops.aten.add.Tensor(input_72, input_73, alpha = 1);  input_72 = input_73 = None
        view_default = torch.ops.aten.view.default(add_tensor, input_74);  add_tensor = input_74 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_75);  input_75 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_76);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_77);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_78, input_79);  input_78 = input_79 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.bfloat16);  view_default_2 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 2, 1, 3]);  convert_element_type_default_2 = None
        slice_tensor = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 256, 1);  permute_default_1 = None
        select_int = torch.ops.aten.select.int(input_80, -1, 1);  input_80 = None
        neg_default = torch.ops.aten.neg.default(input_81);  input_81 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(full_default, neg_default, 3, 1, 9223372036854775807, 2);  neg_default = None
        slice_scatter_default_1 = torch.ops.aten.slice_scatter.default(full_default, select_int, 3, 0, 9223372036854775807, 2);  full_default = select_int = None
        add_tensor_1 = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1, alpha = 1);  slice_scatter_default = slice_scatter_default_1 = None
        clone_default = torch.ops.aten.clone.default(input_82, memory_format = torch.contiguous_format);  input_82 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_83);  clone_default = input_83 = None
        mul_tensor = torch.ops.aten.mul.Tensor(slice_tensor, view_default_3);  slice_tensor = view_default_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor, alpha = 1);  add_tensor_1 = mul_tensor = None
        slice_scatter_default_2 = torch.ops.aten.slice_scatter.default(full_default_1, slice_tensor_1, 3, 64, 9223372036854775807, 1);  slice_tensor_1 = None
        slice_scatter_default_3 = torch.ops.aten.slice_scatter.default(full_default_1, add_tensor_2, 3, 0, 64, 1);  full_default_1 = add_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3, alpha = 1);  slice_scatter_default_2 = slice_scatter_default_3 = None
        view_default_4 = torch.ops.aten.view.default(input_84, input_77);  input_84 = None
        view_default_5 = torch.ops.aten.view.default(input_85, input_77);  input_85 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_3, input_77);  add_tensor_3 = None
        view_default_7 = torch.ops.aten.view.default(view_default_4, input_86);  view_default_4 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default_2, input_76);  permute_default_2 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_7, input_87);  view_default_7 = input_87 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_3, input_77);  mm_default_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_3, alpha = 1);  convert_element_type_default = convert_element_type_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        view_default_9 = torch.ops.aten.view.default(view_default_5, input_86);  view_default_5 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_9, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_3, input_76);  permute_default_3 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_9, input_88);  view_default_9 = input_88 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_5, input_77);  mm_default_5 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_5, alpha = 1);  add_tensor_4 = convert_element_type_default_5 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        view_default_11 = torch.ops.aten.view.default(view_default_6, input_86);  view_default_6 = input_86 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_11, [1, 0])
        mm_default_6 = torch.ops.aten.mm.default(permute_default_4, input_76);  permute_default_4 = input_76 = None
        mm_default_7 = torch.ops.aten.mm.default(view_default_11, input_89);  view_default_11 = input_89 = None
        view_default_12 = torch.ops.aten.view.default(mm_default_7, input_77);  mm_default_7 = input_77 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_12, torch.float32);  view_default_12 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, convert_element_type_default_7, alpha = 1);  add_tensor_5 = convert_element_type_default_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_6, input_90);  input_90 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 4096)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True, dtype = None)
        add_tensor_7 = torch.ops.aten.add.Tensor(input_91, input_92, alpha = 1);  input_91 = input_92 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_7, input_93, alpha = 1);  add_tensor_7 = input_93 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor, input_94);  sub_tensor = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True, dtype = None);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list, alpha = 1);  mul_tensor_2 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_5, alpha = 1);  sub_tensor_1 = mul_tensor_5 = None
        div_tensor = torch.ops.aten.div.Tensor(input_94, 4096);  input_94 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(add_tensor_6, mul_tensor_3);  mul_tensor_3 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], False, dtype = None);  mul_tensor_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(add_tensor_6, [0, 1], False, dtype = None);  add_tensor_6 = None
        return (convert_element_type_default_1, convert_element_type_default_4, convert_element_type_default_6, convert_element_type_default_8, mul_tensor_6, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
