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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59):
        clone_default = torch.ops.aten.clone.default(input_44, memory_format = torch.contiguous_format);  input_44 = None
        _scaled_dot_product_flash_attention_backward_default = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_default, input_45, input_46, input_47, input_48, input_49, None, None, 512, 512, 0.1, False, input_50, input_51, scale = 0.125);  clone_default = input_45 = input_46 = input_47 = input_48 = input_49 = input_50 = input_51 = None
        getitem = _scaled_dot_product_flash_attention_backward_default[0]
        getitem_1 = _scaled_dot_product_flash_attention_backward_default[1]
        getitem_2 = _scaled_dot_product_flash_attention_backward_default[2];  _scaled_dot_product_flash_attention_backward_default = None
        permute_default = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        permute_default_1 = torch.ops.aten.permute.default(getitem_1, [0, 2, 1, 3]);  getitem_1 = None
        permute_default_2 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default = torch.ops.aten.view.default(clone_default_1, input_52);  clone_default_1 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_53);  view_default = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_54);  input_54 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_3, input_55);  permute_default_3 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_52);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_3 = torch.ops.aten.view.default(permute_default_1, input_52);  permute_default_1 = None
        clone_default_2 = torch.ops.aten.clone.default(view_default_3, memory_format = torch.contiguous_format);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(clone_default_2, input_53);  clone_default_2 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, input_56);  input_56 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_4, input_55);  permute_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_52);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_2, alpha = 1);  convert_element_type_default = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        clone_default_3 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_6 = torch.ops.aten.view.default(clone_default_3, input_52);  clone_default_3 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_53);  view_default_6 = input_53 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_7, input_57);  input_57 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_7, [1, 0]);  view_default_7 = None
        mm_default_5 = torch.ops.aten.mm.default(permute_default_5, input_55);  permute_default_5 = input_55 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_4, input_52);  mm_default_4 = input_52 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_4, alpha = 1);  add_tensor = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_1, input_58);  input_58 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_59);  mul_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_1, input_59);  input_59 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], False, dtype = None);  mul_tensor_3 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(add_tensor_1, [0, 1], False, dtype = None);  add_tensor_1 = None
        return (convert_element_type_default_1, convert_element_type_default_3, convert_element_type_default_5, mul_tensor_1, sum_dim_int_list, mul_tensor_2, sum_dim_int_list_1, sum_dim_int_list_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
