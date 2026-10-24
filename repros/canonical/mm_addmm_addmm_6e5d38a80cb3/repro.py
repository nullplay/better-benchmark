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
    def forward(self, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122):
        embedding_default = torch.ops.aten.embedding.default(input_89, input_90, -1, False, False);  input_89 = input_90 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_91);  mul_tensor = input_91 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_92, alpha = 1);  mul_tensor_1 = input_92 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        permute_default = torch.ops.aten.permute.default(input_93, [0, 2, 1, 3]);  input_93 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default = torch.ops.aten.view.default(clone_default, input_94);  clone_default = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_95);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(input_96, [1, 0]);  input_96 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default_1);  view_default_1 = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_94);  mm_default = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_1, input_95);  convert_element_type_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(input_97, [1, 0]);  input_97 = None
        addmm_default = torch.ops.aten.addmm.default(input_98, view_default_3, permute_default_2, beta = 1, alpha = 1);  input_98 = view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default, input_99);  addmm_default = input_99 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_4, 0.5)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(view_default_4, 3.0)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_4, mul_tensor_3, alpha = 1);  view_default_4 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_2, 0.7978845608028654);  add_tensor_2 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_3);  mul_tensor_2 = add_tensor_3 = None
        view_default_5 = torch.ops.aten.view.default(mul_tensor_5, input_100);  mul_tensor_5 = None
        permute_default_3 = torch.ops.aten.permute.default(input_101, [1, 0]);  input_101 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_102, view_default_5, permute_default_3, beta = 1, alpha = 1);  input_102 = view_default_5 = permute_default_3 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_94);  addmm_default_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_2, view_default_6, alpha = 1);  view_default_2 = view_default_6 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, embedding_default, alpha = 1);  add_tensor_4 = embedding_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_3, alpha = 1);  convert_element_type_default_2 = getitem_3 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_103);  mul_tensor_6 = input_103 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_7, input_104, alpha = 1);  mul_tensor_7 = input_104 = None
        view_default_7 = torch.ops.aten.view.default(input_105, input_95);  input_105 = input_95 = None
        permute_default_4 = torch.ops.aten.permute.default(input_106, [1, 0]);  input_106 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_7, permute_default_4);  view_default_7 = permute_default_4 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_1, input_94);  mm_default_1 = None
        view_default_9 = torch.ops.aten.view.default(input_107, input_100);  input_107 = input_100 = None
        permute_default_5 = torch.ops.aten.permute.default(input_108, [1, 0]);  input_108 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_109, view_default_9, permute_default_5, beta = 1, alpha = 1);  input_109 = view_default_9 = permute_default_5 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_2, input_94);  addmm_default_2 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_8, view_default_10, alpha = 1);  view_default_8 = view_default_10 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_8, add_tensor_5, alpha = 1);  add_tensor_8 = add_tensor_5 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_9, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_5, alpha = 1);  convert_element_type_default_3 = getitem_5 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_8, input_110);  mul_tensor_8 = input_110 = None
        mm_default_2 = torch.ops.aten.mm.default(input_111, input_112);  input_111 = input_112 = None
        view_default_11 = torch.ops.aten.view.default(mm_default_2, input_94);  mm_default_2 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_113, input_114, input_115, beta = 1, alpha = 1);  input_113 = input_114 = input_115 = None
        view_default_12 = torch.ops.aten.view.default(addmm_default_3, input_94);  addmm_default_3 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(view_default_11, view_default_12, alpha = 1);  view_default_11 = view_default_12 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(add_tensor_11, add_tensor_9, alpha = 1);  add_tensor_11 = add_tensor_9 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_12, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_7, alpha = 1);  convert_element_type_default_4 = getitem_7 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_116);  sub_tensor_3 = input_116 = None
        view_default_13 = torch.ops.aten.view.default(input_117, input_94);  input_117 = None
        view_default_14 = torch.ops.aten.view.default(input_118, input_94);  input_118 = input_94 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(view_default_13, view_default_14, alpha = 1);  view_default_13 = view_default_14 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(add_tensor_13, add_tensor_12, alpha = 1);  add_tensor_13 = add_tensor_12 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_14, torch.float32)
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [2], correction = 0, keepdim = True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, input_119, alpha = 1);  convert_element_type_default_5 = input_119 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(input_120, input_121, alpha = 1);  input_120 = input_121 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(add_tensor_15, add_tensor_14, alpha = 1);  add_tensor_15 = add_tensor_14 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_16, torch.float32)
        add_tensor_17 = torch.ops.aten.add.Tensor(input_122, add_tensor_16, alpha = 1);  input_122 = add_tensor_16 = None
        return (add_tensor_7, mul_tensor_9, getitem_6, mul_tensor_10, var_mean_correction_4, sub_tensor_4, convert_element_type_default_6, add_tensor_17)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
