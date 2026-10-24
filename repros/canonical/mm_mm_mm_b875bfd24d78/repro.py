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
    def forward(self, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93):
        sub_tensor = torch.ops.aten.sub.Tensor(input_64, input_65, alpha = 1);  input_64 = input_65 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_66, sub_tensor);  input_66 = sub_tensor = None
        permute_default = torch.ops.aten.permute.default(input_67, [0, 2, 1, 3]);  input_67 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default = torch.ops.aten.view.default(input_68, input_69);  input_68 = input_69 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        _scaled_dot_product_flash_attention_backward_default = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_default, permute_default_1, input_70, input_71, input_72, input_73, None, None, 512, 512, 0.1, False, input_74, input_75, scale = 0.125);  clone_default = permute_default_1 = input_70 = input_71 = input_72 = input_73 = input_74 = input_75 = None
        getitem = _scaled_dot_product_flash_attention_backward_default[2];  _scaled_dot_product_flash_attention_backward_default = None
        permute_default_2 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        clone_default_1 = torch.ops.aten.clone.default(input_76, memory_format = torch.contiguous_format);  input_76 = None
        view_default_1 = torch.ops.aten.view.default(clone_default_1, input_77);  clone_default_1 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_78);  input_78 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_3, input_79);  permute_default_3 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_80);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default, alpha = 1);  mul_tensor = convert_element_type_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(clone_default_2, input_81);  clone_default_2 = input_81 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_82, input_83, alpha = 1);  input_82 = input_83 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_1, input_77);  add_tensor_1 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, input_84);  input_84 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_4, input_79);  permute_default_4 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_80);  mm_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_2, alpha = 1);  add_tensor = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convolution_backward_default = torch.ops.aten.convolution_backward.default(input_85, input_86, input_87, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  input_85 = input_86 = input_87 = None
        getitem_1 = convolution_backward_default[0];  convolution_backward_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1]);  convert_element_type_default_4 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, permute_default_5, alpha = 1);  add_tensor_2 = permute_default_5 = None
        view_default_6 = torch.ops.aten.view.default(view_default_3, input_77);  view_default_3 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_6, input_88);  input_88 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_6, [1, 0]);  view_default_6 = None
        mm_default_5 = torch.ops.aten.mm.default(permute_default_6, input_79);  permute_default_6 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_4, input_80);  mm_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, convert_element_type_default_5, alpha = 1);  add_tensor_3 = convert_element_type_default_5 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_5, torch.float32);  mm_default_5 = None
        clone_default_3 = torch.ops.aten.clone.default(input_89, memory_format = torch.contiguous_format);  input_89 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_3, input_77);  clone_default_3 = input_77 = None
        mm_default_6 = torch.ops.aten.mm.default(view_default_8, input_90);  input_90 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_8, [1, 0]);  view_default_8 = None
        mm_default_7 = torch.ops.aten.mm.default(permute_default_7, input_79);  permute_default_7 = input_79 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_6, input_80);  mm_default_6 = input_80 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_7, alpha = 1);  add_tensor_4 = convert_element_type_default_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_7, torch.float32);  mm_default_7 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_91, torch.float32);  input_91 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1.1111111111111112);  convert_element_type_default_9 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_5, mul_tensor_1);  add_tensor_5 = mul_tensor_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_92);  input_92 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_93);  mul_tensor_3 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None);  mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_93, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_4, sum_dim_int_list, alpha = 1);  mul_tensor_4 = sum_dim_int_list = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_93);  input_93 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], False, dtype = None);  mul_tensor_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], False, dtype = None);  mul_tensor_2 = None
        return (convert_element_type_default_1, convert_element_type_default_3, convert_element_type_default_6, convert_element_type_default_8, mul_tensor_6, sub_tensor_1, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
