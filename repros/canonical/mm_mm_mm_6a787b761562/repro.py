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
    def forward(self, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114):
        add_tensor = torch.ops.aten.add.Tensor(input_88, input_89, alpha = 1);  input_88 = input_89 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_90, input_91, alpha = 1);  input_90 = input_91 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor, alpha = 1);  add_tensor_1 = add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_3 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1, alpha = 1);  add_tensor_2 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_92);  mul_tensor = input_92 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_1, input_93, alpha = 1);  mul_tensor_1 = input_93 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_94, torch.bfloat16);  input_94 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_95);  convert_element_type_default_1 = input_95 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_96);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_97, torch.bfloat16);  input_97 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, permute_default_1);  permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_96);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_98, torch.bfloat16);  input_98 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default, permute_default_2);  permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_2, input_96);  mm_default_2 = input_96 = None
        view_default_4 = torch.ops.aten.view.default(view_default_1, input_99);  view_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_2, input_99);  view_default_2 = None
        view_default_6 = torch.ops.aten.view.default(view_default_3, input_99);  view_default_3 = input_99 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        slice_tensor = torch.ops.aten.slice.Tensor(view_default_5, 3, 0, 64, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(view_default_5, 3, 64, 9223372036854775807, 1);  view_default_5 = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(view_default_4, 3, 0, 64, 1)
        slice_tensor_3 = torch.ops.aten.slice.Tensor(view_default_4, 3, 64, 9223372036854775807, 1);  view_default_4 = None
        clone_default = torch.ops.aten.clone.default(input_100, memory_format = torch.contiguous_format);  input_100 = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_101);  clone_default = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_102, 4);  input_102 = None
        expand_default = torch.ops.aten.expand.default(unsqueeze_default, input_103, implicit = False);  unsqueeze_default = input_103 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_8 = torch.ops.aten.view.default(clone_default_1, input_101);  clone_default_1 = input_101 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(slice_tensor, view_default_8)
        slice_tensor_4 = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 9223372036854775807, 2)
        slice_tensor_5 = torch.ops.aten.slice.Tensor(slice_tensor, 3, 1, 9223372036854775807, 2);  slice_tensor = None
        neg_default = torch.ops.aten.neg.default(slice_tensor_5);  slice_tensor_5 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(neg_default, 4);  neg_default = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(slice_tensor_4, 4);  slice_tensor_4 = None
        cat_default = torch.ops.aten.cat.default([unsqueeze_default_1, unsqueeze_default_2], -1);  unsqueeze_default_1 = unsqueeze_default_2 = None
        view_default_9 = torch.ops.aten.view.default(cat_default, input_104);  cat_default = input_104 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_9, view_default_7);  view_default_9 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3, alpha = 1);  mul_tensor_2 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(slice_tensor_2, view_default_8);  view_default_8 = None
        slice_tensor_6 = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 0, 9223372036854775807, 2)
        slice_tensor_7 = torch.ops.aten.slice.Tensor(slice_tensor_2, 3, 1, 9223372036854775807, 2);  slice_tensor_2 = None
        neg_default_1 = torch.ops.aten.neg.default(slice_tensor_7);  slice_tensor_7 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(neg_default_1, 4);  neg_default_1 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(slice_tensor_6, 4);  slice_tensor_6 = None
        cat_default_1 = torch.ops.aten.cat.default([unsqueeze_default_3, unsqueeze_default_4], -1);  unsqueeze_default_3 = unsqueeze_default_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_105, view_default_7);  input_105 = view_default_7 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5, alpha = 1);  mul_tensor_4 = mul_tensor_5 = None
        cat_default_2 = torch.ops.aten.cat.default([add_tensor_5, slice_tensor_1], -1);  add_tensor_5 = slice_tensor_1 = None
        cat_default_3 = torch.ops.aten.cat.default([add_tensor_6, slice_tensor_3], -1);  add_tensor_6 = slice_tensor_3 = None
        permute_default_4 = torch.ops.aten.permute.default(cat_default_2, [0, 2, 1, 3]);  cat_default_2 = None
        permute_default_5 = torch.ops.aten.permute.default(cat_default_3, [0, 2, 1, 3]);  cat_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(permute_default_4, torch.float32);  permute_default_4 = None
        permute_default_6 = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 1, 3, 2]);  convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(permute_default_6, torch.bfloat16);  permute_default_6 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(permute_default_5, torch.bfloat16);  permute_default_5 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_3, input_106, implicit = False);  permute_default_3 = input_106 = None
        view_default_10 = torch.ops.aten.view.default(expand_default_1, input_107);  expand_default_1 = input_107 = None
        bmm_default = torch.ops.aten.bmm.default(input_108, view_default_10);  input_108 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_109, torch.bfloat16);  input_109 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_110, torch.bfloat16);  input_110 = None
        permute_default_7 = torch.ops.aten.permute.default(convert_element_type_default_8, [1, 0]);  convert_element_type_default_8 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_7, view_default, permute_default_7, beta = 1, alpha = 1);  convert_element_type_default_7 = view_default = permute_default_7 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default, input_111);  input_111 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(view_default_11, 0.5)
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_9, 3.0);  convert_element_type_default_9 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_7 = torch.ops.aten.add.Tensor(view_default_11, mul_tensor_7, alpha = 1);  view_default_11 = mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_tensor_7, 0.7978845608028654);  add_tensor_7 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_8);  mul_tensor_8 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_6, add_tensor_8);  mul_tensor_6 = add_tensor_8 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_10, input_112);  convert_element_type_default_10 = input_112 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_113, view_default_12, input_114, beta = 1, alpha = 1);  input_113 = input_114 = None
        permute_default_8 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1]);  view_default_10 = None
        return (cat_default_1, convert_element_type_default_5, convert_element_type_default_6, bmm_default, addmm_default, view_default_12, addmm_default_1, permute_default_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
