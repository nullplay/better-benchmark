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
    def forward(self, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70):
        add_tensor = torch.ops.aten.add.Tensor(input_42, input_43, alpha = 1);  input_42 = input_43 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_44, input_45);  input_44 = input_45 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_46, input_47, alpha = 1);  input_46 = input_47 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_48, sub_tensor);  input_48 = sub_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_49, input_50);  input_49 = input_50 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_51, input_52, alpha = 1);  input_51 = input_52 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_53, sub_tensor_2);  input_53 = sub_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_54, 1024)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_54, [2], True, dtype = None);  input_54 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_55, [2], True, dtype = None);  input_55 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_56, sum_dim_int_list_1);  input_56 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, mul_tensor_5, alpha = 1);  sub_tensor_3 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_57, sub_tensor_4);  input_57 = sub_tensor_4 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_6, alpha = 1);  add_tensor_3 = mul_tensor_6 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_7);  convert_element_type_default = mul_tensor_7 = None
        view_default = torch.ops.aten.view.default(mul_tensor_8, input_59);  mul_tensor_8 = input_59 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_60);  input_60 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_61);  permute_default = input_61 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_62);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_63);  view_default_1 = input_63 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        _scaled_dot_product_flash_attention_backward_default = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_default, input_64, input_65, input_66, input_67, input_68, None, None, 512, 512, 0.1, False, input_69, input_70, scale = 0.125);  clone_default = input_64 = input_65 = input_66 = input_67 = input_68 = input_69 = input_70 = None
        getitem = _scaled_dot_product_flash_attention_backward_default[0]
        getitem_1 = _scaled_dot_product_flash_attention_backward_default[1]
        getitem_2 = _scaled_dot_product_flash_attention_backward_default[2];  _scaled_dot_product_flash_attention_backward_default = None
        permute_default_2 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        permute_default_3 = torch.ops.aten.permute.default(getitem_1, [0, 2, 1, 3]);  getitem_1 = None
        permute_default_4 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        view_default_3 = torch.ops.aten.view.default(permute_default_3, input_62);  permute_default_3 = input_62 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        return (convert_element_type_default_2, clone_default_1, view_default_3, clone_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
