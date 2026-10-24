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
    def forward(self, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76):
        mul_tensor = torch.ops.aten.mul.Tensor(input_50, input_51);  input_50 = input_51 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_52);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_52, sum_dim_int_list_1);  input_52 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_53, sub_tensor_1);  input_53 = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16)
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_54, torch.bfloat16);  input_54 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_5);  convert_element_type_default = mul_tensor_5 = None
        view_default = torch.ops.aten.view.default(mul_tensor_6, input_55);  mul_tensor_6 = input_55 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_56);  input_56 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_57);  permute_default = input_57 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_58);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_59);  view_default_1 = input_59 = None
        slice_tensor = torch.ops.aten.slice.Tensor(view_default_2, 2, 0, 6, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(view_default_2, 2, 6, 12, 1);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(slice_tensor_1, input_60);  slice_tensor_1 = input_60 = None
        permute_default_1 = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1, 3]);  slice_tensor = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(input_61, input_62);  input_61 = input_62 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        _scaled_dot_product_flash_attention_backward_default = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_default, permute_default_2, input_63, input_64, input_65, input_66, None, None, 512, 512, 0.1, False, input_67, input_68, scale = 0.125);  clone_default = permute_default_2 = input_63 = input_64 = input_65 = input_66 = input_67 = input_68 = None
        getitem = _scaled_dot_product_flash_attention_backward_default[0]
        getitem_1 = _scaled_dot_product_flash_attention_backward_default[1]
        getitem_2 = _scaled_dot_product_flash_attention_backward_default[2];  _scaled_dot_product_flash_attention_backward_default = None
        permute_default_3 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        permute_default_4 = torch.ops.aten.permute.default(getitem_1, [0, 2, 1, 3]);  getitem_1 = None
        permute_default_5 = torch.ops.aten.permute.default(getitem_2, [0, 2, 1, 3]);  getitem_2 = None
        clone_default_1 = torch.ops.aten.clone.default(view_default_3, memory_format = torch.contiguous_format);  view_default_3 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_69);  clone_default_1 = input_69 = None
        bmm_default = torch.ops.aten.bmm.default(input_70, view_default_5);  input_70 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_71);  convert_element_type_default_3 = input_71 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(bmm_default, torch.float32);  bmm_default = None
        mm_default_2 = torch.ops.aten.mm.default(input_72, input_73);  input_72 = input_73 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_58);  mm_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_4, convert_element_type_default_5, alpha = 1);  mul_tensor_4 = convert_element_type_default_5 = None
        view_default_7 = torch.ops.aten.view.default(input_74, input_58);  input_74 = input_58 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_6, alpha = 1);  add_tensor = convert_element_type_default_6 = None
        permute_default_6 = torch.ops.aten.permute.default(input_75, [0, 2, 1]);  input_75 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, permute_default_6, alpha = 1);  add_tensor_1 = permute_default_6 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, input_76, alpha = 1);  add_tensor_2 = input_76 = None
        return (convert_element_type_default_2, permute_default_3, permute_default_4, permute_default_5, mul_tensor_7, convert_element_type_default_4, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
