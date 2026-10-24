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
    def forward(self, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93):
        add_tensor = torch.ops.aten.add.Tensor(input_71, input_72, alpha = 1);  input_71 = input_72 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_73);  mul_tensor = input_73 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_74, alpha = 1);  mul_tensor_1 = input_74 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_75)
        permute_default = torch.ops.aten.permute.default(input_76, [1, 0]);  input_76 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_77);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_78);  view_default_1 = None
        slice_tensor = torch.ops.aten.slice.Tensor(view_default_2, 3, 0, 64, 1)
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_79, 4);  input_79 = None
        expand_default = torch.ops.aten.expand.default(unsqueeze_default, input_80, implicit = False);  unsqueeze_default = input_80 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_81);  clone_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(slice_tensor, view_default_3);  view_default_3 = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(slice_tensor, 3, 1, 9223372036854775807, 2)
        neg_default = torch.ops.aten.neg.default(slice_tensor_1);  slice_tensor_1 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(neg_default, 4);  neg_default = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 9223372036854775807, 2);  slice_tensor = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(slice_tensor_2, 4);  slice_tensor_2 = None
        cat_default = torch.ops.aten.cat.default([unsqueeze_default_1, unsqueeze_default_2], -1);  unsqueeze_default_1 = unsqueeze_default_2 = None
        view_default_4 = torch.ops.aten.view.default(cat_default, input_82);  cat_default = input_82 = None
        clone_default_1 = torch.ops.aten.clone.default(input_83, memory_format = torch.contiguous_format);  input_83 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_81);  clone_default_1 = input_81 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_4, view_default_5);  view_default_4 = view_default_5 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3, alpha = 1);  mul_tensor_2 = mul_tensor_3 = None
        slice_tensor_3 = torch.ops.aten.slice.Tensor(view_default_2, 3, 64, 9223372036854775807, 1);  view_default_2 = None
        cat_default_1 = torch.ops.aten.cat.default([add_tensor_3, slice_tensor_3], -1);  add_tensor_3 = slice_tensor_3 = None
        permute_default_1 = torch.ops.aten.permute.default(cat_default_1, [0, 2, 1, 3]);  cat_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(permute_default_1, torch.float32);  permute_default_1 = None
        expand_default_1 = torch.ops.aten.expand.default(convert_element_type_default_2, input_84, implicit = False);  convert_element_type_default_2 = None
        view_default_6 = torch.ops.aten.view.default(expand_default_1, input_85);  expand_default_1 = input_85 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_1, input_75)
        permute_default_2 = torch.ops.aten.permute.default(input_86, [1, 0]);  input_86 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_7, permute_default_2);  view_default_7 = permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_1, input_77);  mm_default_1 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_78);  view_default_8 = None
        slice_tensor_4 = torch.ops.aten.slice.Tensor(view_default_9, 3, 0, 64, 1)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(slice_tensor_4, input_87);  input_87 = None
        slice_tensor_5 = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 1, 9223372036854775807, 2)
        slice_tensor_6 = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 0, 9223372036854775807, 2);  slice_tensor_4 = None
        slice_tensor_7 = torch.ops.aten.slice.Tensor(view_default_9, 3, 64, 9223372036854775807, 1);  view_default_9 = None
        cat_default_2 = torch.ops.aten.cat.default([input_88, slice_tensor_7], -1);  input_88 = slice_tensor_7 = None
        view_default_10 = torch.ops.aten.view.default(convert_element_type_default_1, input_75)
        permute_default_3 = torch.ops.aten.permute.default(input_89, [1, 0]);  input_89 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_10, permute_default_3);  view_default_10 = permute_default_3 = None
        view_default_11 = torch.ops.aten.view.default(mm_default_2, input_77);  mm_default_2 = input_77 = None
        view_default_12 = torch.ops.aten.view.default(view_default_11, input_78);  view_default_11 = input_78 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_12, [0, 2, 1, 3]);  view_default_12 = None
        expand_default_2 = torch.ops.aten.expand.default(permute_default_4, input_84, implicit = False);  permute_default_4 = input_84 = None
        view_default_13 = torch.ops.aten.view.default(convert_element_type_default_1, input_75);  convert_element_type_default_1 = input_75 = None
        permute_default_5 = torch.ops.aten.permute.default(input_90, [1, 0]);  input_90 = None
        addmm_default = torch.ops.aten.addmm.default(input_91, view_default_13, permute_default_5, beta = 1, alpha = 1);  input_91 = view_default_13 = permute_default_5 = None
        view_default_14 = torch.ops.aten.view.default(addmm_default, input_92);  addmm_default = input_92 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_14, 0.5)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(view_default_14, 3.0)
        mul_tensor_6 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_14, mul_tensor_6, alpha = 1);  view_default_14 = mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(add_tensor_4, 0.7978845608028654);  add_tensor_4 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_7);  mul_tensor_7 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_5, add_tensor_5);  mul_tensor_5 = add_tensor_5 = None
        view_default_15 = torch.ops.aten.view.default(mul_tensor_8, input_93);  mul_tensor_8 = input_93 = None
        return (view_default_6, mul_tensor_4, slice_tensor_5, slice_tensor_6, cat_default_2, expand_default_2, view_default_15)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
