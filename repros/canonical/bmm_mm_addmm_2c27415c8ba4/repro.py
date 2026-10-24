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
    def forward(self, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123):
        embedding_default = torch.ops.aten.embedding.default(input_86, input_87, -1, False, False);  input_86 = input_87 = None
        expand_default = torch.ops.aten.expand.default(input_88, input_89, implicit = False);  input_88 = input_89 = None
        view_default = torch.ops.aten.view.default(expand_default, input_90);  expand_default = input_90 = None
        expand_default_1 = torch.ops.aten.expand.default(input_91, input_92, implicit = False);  input_91 = None
        view_default_1 = torch.ops.aten.view.default(expand_default_1, input_93);  expand_default_1 = input_93 = None
        bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default, input_92);  bmm_default = input_92 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_94);  clone_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_95, torch.bfloat16);  input_95 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_96);  view_default_3 = None
        mm_default = torch.ops.aten.mm.default(view_default_4, permute_default_1);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mm_default, input_94);  mm_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_97, 0.5);  input_97 = None
        add_tensor = torch.ops.aten.add.Tensor(input_98, 1.0, alpha = 1);  input_98 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_99, torch.bfloat16);  input_99 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_100, torch.bfloat16);  input_100 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_3, input_101);  convert_element_type_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_6, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_6 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default, input_94);  addmm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_5, view_default_7, alpha = 1);  view_default_5 = view_default_7 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, embedding_default, alpha = 1);  embedding_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_3 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1, alpha = 1)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_102);  mul_tensor_2 = input_102 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_3, input_103, alpha = 1);  mul_tensor_3 = input_103 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        permute_default_3 = torch.ops.aten.permute.default(input_104, [1, 0]);  input_104 = None
        view_default_8 = torch.ops.aten.view.default(input_105, input_96);  input_105 = input_96 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_8, permute_default_3);  view_default_8 = permute_default_3 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_1, input_94);  mm_default_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_106, torch.bfloat16);  input_106 = None
        view_default_10 = torch.ops.aten.view.default(input_107, input_101);  input_107 = input_101 = None
        permute_default_4 = torch.ops.aten.permute.default(input_108, [1, 0]);  input_108 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default_10, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_5 = view_default_10 = permute_default_4 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_1, input_94);  addmm_default_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_9, view_default_11, alpha = 1);  view_default_9 = view_default_11 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, add_tensor_2, alpha = 1);  add_tensor_5 = add_tensor_2 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_6, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_6, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_109);  input_109 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_5, input_110, alpha = 1);  mul_tensor_5 = input_110 = None
        mm_default_2 = torch.ops.aten.mm.default(input_111, input_112);  input_111 = input_112 = None
        view_default_12 = torch.ops.aten.view.default(mm_default_2, input_94);  mm_default_2 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_113, input_114, input_115, beta = 1, alpha = 1);  input_113 = input_114 = input_115 = None
        view_default_13 = torch.ops.aten.view.default(addmm_default_2, input_94);  addmm_default_2 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(view_default_12, view_default_13, alpha = 1);  view_default_12 = view_default_13 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_9, add_tensor_6, alpha = 1);  add_tensor_9 = add_tensor_6 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_10, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_11);  add_tensor_11 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_10, getitem_5, alpha = 1);  getitem_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_116);  input_116 = None
        view_default_14 = torch.ops.aten.view.default(input_117, input_94);  input_117 = None
        view_default_15 = torch.ops.aten.view.default(input_118, input_94);  input_118 = input_94 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(view_default_14, view_default_15, alpha = 1);  view_default_14 = view_default_15 = None
        add_tensor_13 = torch.ops.aten.add.Tensor(add_tensor_12, add_tensor_10, alpha = 1);  add_tensor_12 = add_tensor_10 = None
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(add_tensor_13, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_13, getitem_7, alpha = 1);  getitem_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_119);  sub_tensor_3 = input_119 = None
        add_tensor_14 = torch.ops.aten.add.Tensor(input_120, input_121, alpha = 1);  input_120 = input_121 = None
        add_tensor_15 = torch.ops.aten.add.Tensor(add_tensor_14, add_tensor_13, alpha = 1);  add_tensor_14 = add_tensor_13 = None
        var_mean_correction_4 = torch.ops.aten.var_mean.correction(add_tensor_15, [2], correction = 0, keepdim = True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_15, input_122, alpha = 1);  input_122 = None
        add_tensor_16 = torch.ops.aten.add.Tensor(input_123, add_tensor_15, alpha = 1);  input_123 = add_tensor_15 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_1, 4096);  rsqrt_default_1 = None
        return (view_default_4, add_tensor_1, getitem_1, rsqrt_default, convert_element_type_default_4, mul_tensor_4, add_tensor_8, rsqrt_default_2, mul_tensor_6, mul_tensor_7, getitem_6, mul_tensor_8, var_mean_correction_4, sub_tensor_4, add_tensor_16, div_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
