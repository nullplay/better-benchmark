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
    def forward(self, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113):
        add_tensor = torch.ops.aten.add.Tensor(input_81, input_82, alpha = 1);  input_81 = input_82 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_83, 1.1111111111111112);  input_83 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, add_tensor, alpha = 1);  mul_tensor = add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_84);  mul_tensor_1 = input_84 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_2, input_85, alpha = 1);  mul_tensor_2 = input_85 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_86, torch.bfloat16);  input_86 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_87, torch.bfloat16);  input_87 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_88);  convert_element_type_default_2 = input_88 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_89);  addmm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_90, torch.bfloat16);  input_90 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_91, torch.bfloat16);  input_91 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default_1, input_89);  addmm_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(add_tensor_3, [0, 2, 1])
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_92, torch.bfloat16);  input_92 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(permute_default_2, torch.bfloat16);  permute_default_2 = None
        convolution_default = torch.ops.aten.convolution.default(convert_element_type_default_6, convert_element_type_default_5, None, [1], [4], [1], False, [0], 768);  convert_element_type_default_6 = convert_element_type_default_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_93, torch.bfloat16);  input_93 = None
        convolution_default_1 = torch.ops.aten.convolution.default(convolution_default, convert_element_type_default_7, None, [1], [0], [1], False, [0], 1);  convolution_default = convert_element_type_default_7 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(convolution_default_1, input_94, alpha = 1);  convolution_default_1 = input_94 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_95, torch.bfloat16);  input_95 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_96, torch.bfloat16);  input_96 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0]);  convert_element_type_default_10 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_9, view_default, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_9 = permute_default_3 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_2, input_89)
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_97)
        view_default_5 = torch.ops.aten.view.default(view_default_1, input_97);  view_default_1 = None
        view_default_6 = torch.ops.aten.view.default(view_default_2, input_97);  view_default_2 = input_97 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_8, [0, 2, 1]);  convert_element_type_default_8 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(permute_default_4, view_default_3);  permute_default_4 = view_default_3 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(input_98, torch.bfloat16);  input_98 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(input_99, torch.bfloat16);  input_99 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_12, [1, 0]);  convert_element_type_default_12 = None
        clone_default = torch.ops.aten.clone.default(mul_tensor_3, memory_format = torch.contiguous_format);  mul_tensor_3 = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_100);  clone_default = input_100 = None
        mm_default = torch.ops.aten.mm.default(view_default_7, permute_default_5);  permute_default_5 = None
        view_default_8 = torch.ops.aten.view.default(mm_default, input_101);  input_101 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_8, convert_element_type_default_11, alpha = 1);  view_default_8 = convert_element_type_default_11 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_5, input_102);  add_tensor_5 = input_102 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(input_103, torch.bfloat16);  input_103 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(input_104, torch.bfloat16);  input_104 = None
        permute_default_6 = torch.ops.aten.permute.default(convert_element_type_default_14, [1, 0]);  convert_element_type_default_14 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_13, view_default, permute_default_6, beta = 1, alpha = 1);  convert_element_type_default_13 = view_default = permute_default_6 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_3, input_89);  addmm_default_3 = input_89 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1]);  view_default_10 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(clone_default_1, -1);  clone_default_1 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(unsqueeze_default, [0, 0, 4, 4], 0.0);  unsqueeze_default = None
        index_tensor = torch.ops.aten.index.Tensor(constant_pad_nd_default, [None, None, input_105, input_106]);  constant_pad_nd_default = input_105 = input_106 = None
        permute_default_8 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        permute_default_9 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        permute_default_10 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_107, input_108, input_109, beta = 1, alpha = 1);  input_107 = input_108 = input_109 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_4, input_110);  addmm_default_4 = input_110 = None
        convert_element_type_default_15 = torch.ops.prims.convert_element_type.default(input_111, torch.bfloat16);  input_111 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_15, 0.1);  convert_element_type_default_15 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(gt_scalar, view_default_11);  gt_scalar = view_default_11 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_5, add_tensor_3, alpha = 1);  mul_tensor_5 = add_tensor_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_6, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_2, 1e-12, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_6, getitem_3, alpha = 1);  add_tensor_6 = getitem_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_112);  input_112 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_7, input_113, alpha = 1);  mul_tensor_7 = input_113 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_1, 768);  rsqrt_default_1 = None
        return (addmm_default_2, view_default_7, mm_default, view_default_9, index_tensor, permute_default_8, permute_default_9, permute_default_10, mul_tensor_6, add_tensor_8, div_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
