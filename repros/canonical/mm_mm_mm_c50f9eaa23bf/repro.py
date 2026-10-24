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
    def forward(self, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100):
        full_default = torch.ops.aten.full.default(input_73, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_73 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_74, input_75);  input_74 = input_75 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_76, 0.7978845608028654);  input_76 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_77, input_78);  input_77 = input_78 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default_1, convert_element_type_default_2, alpha = 1);  convert_element_type_default_1 = convert_element_type_default_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default, 0.5);  convert_element_type_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_3, alpha = 1);  add_tensor = mul_tensor_3 = None
        view_default = torch.ops.aten.view.default(add_tensor_1, input_79);  add_tensor_1 = input_79 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_80);  input_80 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_81);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_82);  mm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_83, input_84);  input_83 = input_84 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_85, torch.bfloat16);  input_85 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1, 3]);  convert_element_type_default_5 = None
        slice_tensor = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 256, 1);  permute_default_1 = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(input_86, 3, 64, 256, 1);  input_86 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(input_87, input_88, 3, 1, 9223372036854775807, 2);  input_88 = None
        slice_scatter_default_1 = torch.ops.aten.slice_scatter.default(input_87, input_89, 3, 0, 9223372036854775807, 2);  input_87 = input_89 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1, alpha = 1);  slice_scatter_default = slice_scatter_default_1 = None
        view_default_3 = torch.ops.aten.view.default(input_90, input_91);  input_90 = input_91 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(slice_tensor, view_default_3);  slice_tensor = view_default_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_4, alpha = 1);  add_tensor_2 = mul_tensor_4 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(input_92, input_93, alpha = 1);  input_92 = input_93 = None
        slice_scatter_default_2 = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_1, 3, 64, 9223372036854775807, 1);  slice_tensor_1 = None
        slice_scatter_default_3 = torch.ops.aten.slice_scatter.default(full_default, add_tensor_3, 3, 0, 64, 1);  add_tensor_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3, alpha = 1);  slice_scatter_default_2 = slice_scatter_default_3 = None
        slice_scatter_default_4 = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_2, 3, 64, 9223372036854775807, 1);  slice_tensor_2 = None
        slice_scatter_default_5 = torch.ops.aten.slice_scatter.default(full_default, add_tensor_4, 3, 0, 64, 1);  full_default = add_tensor_4 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(slice_scatter_default_4, slice_scatter_default_5, alpha = 1);  slice_scatter_default_4 = slice_scatter_default_5 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_82);  clone_default = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_6, input_82);  add_tensor_6 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_5, input_82);  add_tensor_5 = None
        view_default_7 = torch.ops.aten.view.default(view_default_4, input_94);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default_3, input_81);  permute_default_3 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_7, input_95);  view_default_7 = input_95 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_3, input_82);  mm_default_3 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(convert_element_type_default_3, convert_element_type_default_6, alpha = 1);  convert_element_type_default_3 = convert_element_type_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        view_default_9 = torch.ops.aten.view.default(view_default_5, input_94);  view_default_5 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_9, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_4, input_81);  permute_default_4 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_9, input_96);  view_default_9 = input_96 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_5, input_82);  mm_default_5 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_7, convert_element_type_default_8, alpha = 1);  add_tensor_7 = convert_element_type_default_8 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        view_default_11 = torch.ops.aten.view.default(view_default_6, input_94);  view_default_6 = input_94 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_11, [1, 0])
        mm_default_6 = torch.ops.aten.mm.default(permute_default_5, input_81);  permute_default_5 = input_81 = None
        mm_default_7 = torch.ops.aten.mm.default(view_default_11, input_97);  view_default_11 = input_97 = None
        view_default_12 = torch.ops.aten.view.default(mm_default_7, input_82);  mm_default_7 = input_82 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_12, torch.float32);  view_default_12 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_8, convert_element_type_default_10, alpha = 1);  add_tensor_8 = convert_element_type_default_10 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_9, input_98);  input_98 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 4096)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_99, input_100);  input_99 = input_100 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_7);  mul_tensor_5 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor_9, mul_tensor_7);  mul_tensor_7 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1], False, dtype = None);  mul_tensor_9 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(add_tensor_9, [0, 1], False, dtype = None);  add_tensor_9 = None
        return (convert_element_type_default_4, convert_element_type_default_7, convert_element_type_default_9, convert_element_type_default_11, mul_tensor_6, sum_dim_int_list, mul_tensor_8, sum_dim_int_list_1, sum_dim_int_list_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
