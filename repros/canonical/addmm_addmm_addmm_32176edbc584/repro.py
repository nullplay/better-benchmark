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
    def forward(self, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97):
        add_tensor = torch.ops.aten.add.Tensor(input_69, input_70, alpha = 1);  input_69 = input_70 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_71);  mul_tensor = input_71 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_72, alpha = 1);  mul_tensor_1 = input_72 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_73)
        permute_default = torch.ops.aten.permute.default(input_74, [1, 0]);  input_74 = None
        addmm_default = torch.ops.aten.addmm.default(input_75, view_default, permute_default, beta = 1, alpha = 1);  input_75 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_76);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_77)
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_1, input_73)
        permute_default_2 = torch.ops.aten.permute.default(input_78, [1, 0]);  input_78 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_79, view_default_3, permute_default_2, beta = 1, alpha = 1);  input_79 = view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_76);  addmm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_77);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_1, input_73)
        permute_default_4 = torch.ops.aten.permute.default(input_80, [1, 0]);  input_80 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_81, view_default_6, permute_default_4, beta = 1, alpha = 1);  input_81 = view_default_6 = permute_default_4 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_2, input_76);  addmm_default_2 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_77);  view_default_7 = input_77 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_1, input_73)
        permute_default_6 = torch.ops.aten.permute.default(input_82, [1, 0]);  input_82 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_83, view_default_9, permute_default_6, beta = 1, alpha = 1);  input_83 = view_default_9 = permute_default_6 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_3, input_76);  addmm_default_3 = input_76 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1]);  view_default_10 = None
        clone_default = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(clone_default, -1);  clone_default = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(unsqueeze_default, [0, 0, 4, 4], 0.0);  unsqueeze_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_84, -1);  input_84 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, -1);  unsqueeze_default_1 = None
        full_default = torch.ops.aten.full.default(input_85, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_85 = None
        index_tensor = torch.ops.aten.index.Tensor(constant_pad_nd_default, [None, None, unsqueeze_default_2, full_default]);  constant_pad_nd_default = unsqueeze_default_2 = full_default = None
        permute_default_8 = torch.ops.aten.permute.default(index_tensor, [0, 1, 2, 4, 3, 5]);  index_tensor = None
        view_default_11 = torch.ops.aten.view.default(permute_default_8, input_86);  permute_default_8 = input_86 = None
        permute_default_9 = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 2, 1])
        convolution_default = torch.ops.aten.convolution.default(permute_default_9, input_87, None, [1], [4], [1], False, [0], 768);  permute_default_9 = input_87 = None
        convolution_default_1 = torch.ops.aten.convolution.default(convolution_default, input_88, None, [1], [0], [1], False, [0], 1);  convolution_default = input_88 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(convolution_default_1, input_89, alpha = 1);  convolution_default_1 = input_89 = None
        permute_default_10 = torch.ops.aten.permute.default(add_tensor_3, [0, 2, 1]);  add_tensor_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(permute_default_10, view_default_1);  permute_default_10 = view_default_1 = None
        clone_default_1 = torch.ops.aten.clone.default(mul_tensor_2, memory_format = torch.contiguous_format);  mul_tensor_2 = None
        view_default_12 = torch.ops.aten.view.default(clone_default_1, input_90);  clone_default_1 = input_90 = None
        cat_default = torch.ops.aten.cat.default([input_91, input_92], 2);  input_91 = input_92 = None
        view_default_13 = torch.ops.aten.view.default(cat_default, input_93);  cat_default = None
        view_default_14 = torch.ops.aten.view.default(view_default_13, input_73);  view_default_13 = input_73 = None
        permute_default_11 = torch.ops.aten.permute.default(input_94, [1, 0]);  input_94 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_95, view_default_14, permute_default_11, beta = 1, alpha = 1);  input_95 = view_default_14 = permute_default_11 = None
        view_default_15 = torch.ops.aten.view.default(addmm_default_4, input_93);  addmm_default_4 = input_93 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_15, convert_element_type_default_1, alpha = 1);  view_default_15 = convert_element_type_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32);  add_tensor_4 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_3, alpha = 1);  convert_element_type_default_2 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-12, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_96);  mul_tensor_3 = input_96 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_4, input_97, alpha = 1);  mul_tensor_4 = input_97 = None
        return (permute_default_1, permute_default_3, permute_default_5, view_default_11, view_default_12, add_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
