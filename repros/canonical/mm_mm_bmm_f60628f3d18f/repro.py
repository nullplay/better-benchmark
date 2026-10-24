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
    def forward(self, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131, input_132, input_133, input_134, input_135, input_136, input_137):
        view_default = torch.ops.aten.view.default(input_108, input_109);  input_108 = input_109 = None
        permute_default = torch.ops.aten.permute.default(input_110, [1, 0]);  input_110 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_111);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_112);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_113, implicit = False);  permute_default_1 = input_113 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_114);  clone_default = None
        mm_default_1 = torch.ops.aten.mm.default(input_115, input_116);  input_115 = input_116 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_111);  mm_default_1 = input_111 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_112);  view_default_4 = input_112 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 2]);  permute_default_2 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_3, input_117, implicit = False);  permute_default_3 = input_117 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_118);  clone_default_1 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_3, view_default_6);  view_default_3 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_119);  bmm_default = None
        where_self = torch.ops.aten.where.self(input_120, input_121, input_122);  input_120 = input_121 = input_122 = None
        add_tensor = torch.ops.aten.add.Tensor(where_self, 0, alpha = 1);  where_self = None
        embedding_default = torch.ops.aten.embedding.default(input_123, add_tensor, -1, False, False);  input_123 = add_tensor = None
        permute_default_4 = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(permute_default_4, 0);  permute_default_4 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_124, 1);  input_124 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 2);  unsqueeze_default_1 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(input_125, 1);  input_125 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 3);  unsqueeze_default_3 = None
        le_tensor = torch.ops.aten.le.Tensor(unsqueeze_default_2, unsqueeze_default_4);  unsqueeze_default_2 = unsqueeze_default_4 = None
        expand_default_2 = torch.ops.aten.expand.default(le_tensor, input_126, implicit = False);  le_tensor = input_126 = None
        full_default = torch.ops.aten.full.default(input_127, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default(input_127, -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_127 = None
        where_self_1 = torch.ops.aten.where.self(expand_default_2, full_default, full_default_1);  expand_default_2 = full_default = full_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(unsqueeze_default, where_self_1, alpha = 1);  unsqueeze_default = where_self_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_7, add_tensor_1, alpha = 1);  view_default_7 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default, alpha = 1);  convert_element_type_default = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        expand_default_3 = torch.ops.aten.expand.default(convert_element_type_default_1, input_119, implicit = False);  convert_element_type_default_1 = None
        clone_default_2 = torch.ops.aten.clone.default(input_128, memory_format = torch.contiguous_format);  input_128 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_2, input_114);  clone_default_2 = None
        clone_default_3 = torch.ops.aten.clone.default(input_129, memory_format = torch.contiguous_format);  input_129 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_3, input_118);  clone_default_3 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_8, view_default_9);  view_default_8 = view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(bmm_default_1, input_119);  bmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_10, add_tensor_1, alpha = 1);  view_default_10 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32);  add_tensor_3 = None
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default_2, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default_1, alpha = 1);  convert_element_type_default_2 = amax_default_1 = None
        exp_default_1 = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(exp_default_1, [-1], True, dtype = None)
        div_tensor_1 = torch.ops.aten.div.Tensor(exp_default_1, sum_dim_int_list_1);  exp_default_1 = sum_dim_int_list_1 = None
        clone_default_4 = torch.ops.aten.clone.default(input_130, memory_format = torch.contiguous_format);  input_130 = None
        view_default_11 = torch.ops.aten.view.default(clone_default_4, input_114);  clone_default_4 = None
        clone_default_5 = torch.ops.aten.clone.default(input_131, memory_format = torch.contiguous_format);  input_131 = None
        view_default_12 = torch.ops.aten.view.default(clone_default_5, input_118);  clone_default_5 = None
        bmm_default_2 = torch.ops.aten.bmm.default(view_default_11, view_default_12);  view_default_11 = view_default_12 = None
        view_default_13 = torch.ops.aten.view.default(bmm_default_2, input_119);  bmm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_13, add_tensor_1, alpha = 1);  view_default_13 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32);  add_tensor_4 = None
        amax_default_2 = torch.ops.aten.amax.default(convert_element_type_default_3, [-1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, amax_default_2, alpha = 1);  convert_element_type_default_3 = amax_default_2 = None
        exp_default_2 = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(exp_default_2, [-1], True, dtype = None)
        div_tensor_2 = torch.ops.aten.div.Tensor(exp_default_2, sum_dim_int_list_2);  exp_default_2 = sum_dim_int_list_2 = None
        clone_default_6 = torch.ops.aten.clone.default(input_132, memory_format = torch.contiguous_format);  input_132 = None
        view_default_14 = torch.ops.aten.view.default(clone_default_6, input_114);  clone_default_6 = None
        clone_default_7 = torch.ops.aten.clone.default(input_133, memory_format = torch.contiguous_format);  input_133 = None
        view_default_15 = torch.ops.aten.view.default(clone_default_7, input_118);  clone_default_7 = None
        bmm_default_3 = torch.ops.aten.bmm.default(view_default_14, view_default_15);  view_default_14 = view_default_15 = None
        view_default_16 = torch.ops.aten.view.default(bmm_default_3, input_119);  bmm_default_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_16, add_tensor_1, alpha = 1);  view_default_16 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32);  add_tensor_5 = None
        amax_default_3 = torch.ops.aten.amax.default(convert_element_type_default_4, [-1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, amax_default_3, alpha = 1);  convert_element_type_default_4 = amax_default_3 = None
        exp_default_3 = torch.ops.aten.exp.default(sub_tensor_3);  sub_tensor_3 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(exp_default_3, [-1], True, dtype = None)
        div_tensor_3 = torch.ops.aten.div.Tensor(exp_default_3, sum_dim_int_list_3);  exp_default_3 = sum_dim_int_list_3 = None
        clone_default_8 = torch.ops.aten.clone.default(input_134, memory_format = torch.contiguous_format);  input_134 = None
        view_default_17 = torch.ops.aten.view.default(clone_default_8, input_114);  clone_default_8 = None
        clone_default_9 = torch.ops.aten.clone.default(input_135, memory_format = torch.contiguous_format);  input_135 = None
        view_default_18 = torch.ops.aten.view.default(clone_default_9, input_118);  clone_default_9 = None
        bmm_default_4 = torch.ops.aten.bmm.default(view_default_17, view_default_18);  view_default_17 = view_default_18 = None
        view_default_19 = torch.ops.aten.view.default(bmm_default_4, input_119);  bmm_default_4 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_19, add_tensor_1, alpha = 1);  view_default_19 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32);  add_tensor_6 = None
        amax_default_4 = torch.ops.aten.amax.default(convert_element_type_default_5, [-1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, amax_default_4, alpha = 1);  convert_element_type_default_5 = amax_default_4 = None
        exp_default_4 = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(exp_default_4, [-1], True, dtype = None)
        div_tensor_4 = torch.ops.aten.div.Tensor(exp_default_4, sum_dim_int_list_4);  exp_default_4 = sum_dim_int_list_4 = None
        clone_default_10 = torch.ops.aten.clone.default(input_136, memory_format = torch.contiguous_format);  input_136 = None
        view_default_20 = torch.ops.aten.view.default(clone_default_10, input_114);  clone_default_10 = input_114 = None
        clone_default_11 = torch.ops.aten.clone.default(input_137, memory_format = torch.contiguous_format);  input_137 = None
        view_default_21 = torch.ops.aten.view.default(clone_default_11, input_118);  clone_default_11 = input_118 = None
        bmm_default_5 = torch.ops.aten.bmm.default(view_default_20, view_default_21);  view_default_20 = view_default_21 = None
        view_default_22 = torch.ops.aten.view.default(bmm_default_5, input_119);  bmm_default_5 = input_119 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(view_default_22, add_tensor_1, alpha = 1);  view_default_22 = add_tensor_1 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.float32);  add_tensor_7 = None
        amax_default_5 = torch.ops.aten.amax.default(convert_element_type_default_6, [-1], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_6, amax_default_5, alpha = 1);  convert_element_type_default_6 = amax_default_5 = None
        exp_default_5 = torch.ops.aten.exp.default(sub_tensor_5);  sub_tensor_5 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(exp_default_5, [-1], True, dtype = None)
        div_tensor_5 = torch.ops.aten.div.Tensor(exp_default_5, sum_dim_int_list_5);  exp_default_5 = sum_dim_int_list_5 = None
        return (expand_default_3, div_tensor_1, div_tensor_2, div_tensor_3, div_tensor_4, div_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
