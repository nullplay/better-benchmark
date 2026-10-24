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
    def forward(self, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126):
        mul_tensor = torch.ops.aten.mul.Tensor(input_86, input_87);  input_86 = input_87 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 144)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(input_88, input_89, alpha = 1);  input_88 = input_89 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_90);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4, alpha = 1);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor = torch.ops.aten.div.Tensor(input_90, 144);  input_90 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_91)
        mm_default = torch.ops.aten.mm.default(view_default, input_92);  input_92 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_93);  permute_default = input_93 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_94);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_95);  convert_element_type_default_2 = input_95 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_96, input_97);  input_96 = input_97 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 144)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True, dtype = None)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(input_98, input_99, alpha = 1);  input_98 = input_99 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_100);  sub_tensor_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_7, mul_tensor_9);  mul_tensor_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [2], True, dtype = None);  mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_9, sum_dim_int_list_3);  mul_tensor_9 = sum_dim_int_list_3 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_8, sum_dim_int_list_2, alpha = 1);  mul_tensor_8 = sum_dim_int_list_2 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_11, alpha = 1);  sub_tensor_4 = mul_tensor_11 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(input_100, 144);  input_100 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_5);  div_tensor_1 = sub_tensor_5 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_3, alpha = 1);  convert_element_type_default = convert_element_type_default_3 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor, input_91)
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_101);  input_101 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_102);  permute_default_1 = input_102 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_2, input_103);  mm_default_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_104);  view_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        full_default = torch.ops.aten.full.default(input_105, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_105 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(full_default, permute_default_2, -1, 0, 36, 1);  full_default = permute_default_2 = None
        _scaled_dot_product_flash_attention_backward_default = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(slice_scatter_default, input_106, input_107, input_108, input_109, input_110, None, None, 256, 256, 0.0, False, input_111, input_112, scale = 0.16666666666666666);  slice_scatter_default = input_106 = input_107 = input_108 = input_109 = input_110 = input_111 = input_112 = None
        getitem = _scaled_dot_product_flash_attention_backward_default[0]
        getitem_1 = _scaled_dot_product_flash_attention_backward_default[1]
        getitem_2 = _scaled_dot_product_flash_attention_backward_default[2];  _scaled_dot_product_flash_attention_backward_default = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(getitem_2, [0, -4], 0);  getitem_2 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(getitem_1, [0, -4], 0);  getitem_1 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(getitem, [0, -4], 0);  getitem = None
        cat_default = torch.ops.aten.cat.default([constant_pad_nd_default_2, constant_pad_nd_default_1, constant_pad_nd_default], 0);  constant_pad_nd_default_2 = constant_pad_nd_default_1 = constant_pad_nd_default = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_113, 144)
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(input_113, [2], True, dtype = None);  input_113 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_114, input_115);  input_114 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(input_116, [2], True, dtype = None);  input_116 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_14, sum_dim_int_list_5);  mul_tensor_14 = sum_dim_int_list_5 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(mul_tensor_13, sum_dim_int_list_4, alpha = 1);  mul_tensor_13 = sum_dim_int_list_4 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(sub_tensor_6, mul_tensor_15, alpha = 1);  sub_tensor_6 = mul_tensor_15 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(input_115, 144);  input_115 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(div_tensor_2, sub_tensor_7);  div_tensor_2 = sub_tensor_7 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.bfloat16);  mul_tensor_16 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_5, alpha = 1);  add_tensor = convert_element_type_default_5 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_1, input_91)
        mm_default_4 = torch.ops.aten.mm.default(view_default_5, input_117);  input_117 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [1, 0]);  view_default_5 = None
        mm_default_5 = torch.ops.aten.mm.default(permute_default_3, input_118);  permute_default_3 = input_118 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_4, input_94);  mm_default_4 = input_94 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, input_119);  convert_element_type_default_7 = input_119 = None
        mul_tensor_18 = torch.ops.aten.mul.Tensor(input_120, input_121);  input_120 = input_121 = None
        sub_tensor_8 = torch.ops.aten.sub.Tensor(input_122, input_123, alpha = 1);  input_122 = input_123 = None
        sub_tensor_9 = torch.ops.aten.sub.Tensor(sub_tensor_8, mul_tensor_18, alpha = 1);  sub_tensor_8 = mul_tensor_18 = None
        div_tensor_3 = torch.ops.aten.div.Tensor(input_124, 144);  input_124 = None
        mul_tensor_19 = torch.ops.aten.mul.Tensor(div_tensor_3, sub_tensor_9);  div_tensor_3 = sub_tensor_9 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mul_tensor_19, torch.bfloat16);  mul_tensor_19 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_8, alpha = 1);  add_tensor_1 = convert_element_type_default_8 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_2, input_91);  add_tensor_2 = input_91 = None
        mm_default_6 = torch.ops.aten.mm.default(view_default_7, input_125);  input_125 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_7, [1, 0]);  view_default_7 = None
        mm_default_7 = torch.ops.aten.mm.default(permute_default_4, input_126);  permute_default_4 = input_126 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_6, input_103);  mm_default_6 = input_103 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mm_default_7, torch.float32);  mm_default_7 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_104);  view_default_8 = input_104 = None
        return (convert_element_type_default_1, mul_tensor_6, convert_element_type_default_4, cat_default, convert_element_type_default_6, mul_tensor_17, convert_element_type_default_9, view_default_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
