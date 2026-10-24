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
    def forward(self, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101):
        permute_default = torch.ops.aten.permute.default(input_65, [0, 2, 1, 3]);  input_65 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default = torch.ops.aten.view.default(input_66, input_67);  input_66 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_68);  input_68 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        _scaled_dot_product_flash_attention_backward_default = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_default, permute_default_1, input_69, input_70, input_71, input_72, None, None, 512, 512, 0.1, False, input_73, input_74, scale = 0.125);  clone_default = permute_default_1 = input_69 = input_70 = input_71 = input_72 = input_73 = input_74 = None
        getitem = _scaled_dot_product_flash_attention_backward_default[0];  _scaled_dot_product_flash_attention_backward_default = None
        permute_default_2 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default_2 = torch.ops.aten.view.default(input_75, input_76);  input_75 = input_76 = None
        bmm_default = torch.ops.aten.bmm.default(input_77, view_default_2);  input_77 = view_default_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_78, input_79, alpha = 1);  input_78 = input_79 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_80);  exp_default = input_80 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(bmm_default, torch.float32);  bmm_default = None
        mm_default = torch.ops.aten.mm.default(input_81, input_82);  input_81 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, div_tensor);  convert_element_type_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_1, input_83);  convert_element_type_default_1 = input_83 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_3, [0, 1], True, dtype = torch.float32)
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_1, input_84);  sum_dim_int_list_1 = input_84 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.bfloat16);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(view_default_3, input_85);  view_default_3 = input_85 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default_3, input_86);  permute_default_3 = input_86 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_5, input_87);  view_default_5 = input_87 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_67);  mm_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(input_88, input_89, alpha = 1);  input_88 = input_89 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1]);  convert_element_type_default_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default_6, permute_default_4);  permute_default_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_6, view_default);  view_default_6 = view_default = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_1, input_67);  clone_default_1 = input_67 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, view_default_7, alpha = 1);  mul_tensor_1 = view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_1, input_90);  add_tensor_1 = input_90 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_8, input_91);  input_91 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_5, input_82);  permute_default_5 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_8, [0], True, dtype = torch.float32);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(sum_dim_int_list_2, input_92);  sum_dim_int_list_2 = input_92 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_9, torch.bfloat16);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_3, input_93);  mm_default_3 = input_93 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_94, convert_element_type_default_7, alpha = 1);  input_94 = convert_element_type_default_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(convert_element_type_default_6, torch.float32);  convert_element_type_default_6 = None
        permute_default_6 = torch.ops.aten.permute.default(mul_tensor_2, [0, 2, 1]);  mul_tensor_2 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(permute_default_6, [0, 2], True, dtype = torch.float32)
        view_default_11 = torch.ops.aten.view.default(sum_dim_int_list_3, input_95);  sum_dim_int_list_3 = input_95 = None
        convolution_backward_default = torch.ops.aten.convolution_backward.default(permute_default_6, input_96, input_97, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_default_6 = input_96 = input_97 = None
        getitem_1 = convolution_backward_default[0]
        getitem_2 = convolution_backward_default[1];  convolution_backward_default = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(getitem_2, torch.float32);  getitem_2 = None
        convolution_backward_default_1 = torch.ops.aten.convolution_backward.default(getitem_1, input_98, input_99, [0], [1], [4], [1], False, [0], 768, [True, True, False]);  getitem_1 = input_98 = input_99 = None
        getitem_3 = convolution_backward_default_1[0]
        getitem_4 = convolution_backward_default_1[1];  convolution_backward_default_1 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(getitem_4, torch.float32);  getitem_4 = None
        mm_default_5 = torch.ops.aten.mm.default(input_100, input_82);  input_100 = None
        mm_default_6 = torch.ops.aten.mm.default(input_101, input_82);  input_101 = input_82 = None
        return (mm_default, convert_element_type_default_3, convert_element_type_default_4, add_tensor_2, convert_element_type_default_8, convert_element_type_default_9, view_default_11, convert_element_type_default_10, convert_element_type_default_11, convert_element_type_default_12, mm_default_5, mm_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
