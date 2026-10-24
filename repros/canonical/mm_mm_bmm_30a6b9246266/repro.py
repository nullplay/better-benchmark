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
    def forward(self, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119, input_120, input_121, input_122, input_123, input_124, input_125, input_126, input_127, input_128, input_129, input_130, input_131):
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_104, 1);  input_104 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 3)
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        le_tensor = torch.ops.aten.le.Tensor(unsqueeze_default_2, unsqueeze_default_1);  unsqueeze_default_2 = unsqueeze_default_1 = None
        expand_default = torch.ops.aten.expand.default(le_tensor, input_105, implicit = False);  le_tensor = input_105 = None
        full_default = torch.ops.aten.full.default(input_106, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default(input_106, -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_106 = None
        where_self = torch.ops.aten.where.self(expand_default, full_default, full_default_1);  expand_default = full_default = full_default_1 = None
        permute_default = torch.ops.aten.permute.default(input_107, [1, 0]);  input_107 = None
        view_default = torch.ops.aten.view.default(input_108, input_109);  input_108 = input_109 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_110);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_111);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_112);  view_default = input_112 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_110);  mm_default_1 = input_110 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_111);  view_default_3 = input_111 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 2]);  permute_default_2 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_1, input_113, implicit = False);  permute_default_1 = input_113 = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_114);  clone_default = None
        expand_default_2 = torch.ops.aten.expand.default(permute_default_3, input_115, implicit = False);  permute_default_3 = input_115 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_116);  clone_default_1 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_5, view_default_6)
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_117)
        where_self_1 = torch.ops.aten.where.self(input_118, input_119, input_120);  input_118 = input_119 = input_120 = None
        add_tensor = torch.ops.aten.add.Tensor(where_self_1, 0, alpha = 1);  where_self_1 = None
        embedding_default = torch.ops.aten.embedding.default(input_121, add_tensor, -1, False, False);  input_121 = add_tensor = None
        permute_default_4 = torch.ops.aten.permute.default(embedding_default, [2, 0, 1]);  embedding_default = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(permute_default_4, 0);  permute_default_4 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(unsqueeze_default_3, where_self, alpha = 1);  unsqueeze_default_3 = where_self = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_7, add_tensor_1, alpha = 1);  view_default_7 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_1, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax_default, alpha = 1);  convert_element_type_default_1 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        clone_default_2 = torch.ops.aten.clone.default(input_122, memory_format = torch.contiguous_format);  input_122 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_2, input_114);  clone_default_2 = None
        clone_default_3 = torch.ops.aten.clone.default(input_123, memory_format = torch.contiguous_format);  input_123 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_3, input_116);  clone_default_3 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_8, view_default_9);  view_default_8 = view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(bmm_default_1, input_117);  bmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_10, add_tensor_1, alpha = 1);  view_default_10 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float32)
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default_4, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, amax_default_1, alpha = 1);  convert_element_type_default_4 = None
        exp_default_1 = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        clone_default_4 = torch.ops.aten.clone.default(input_124, memory_format = torch.contiguous_format);  input_124 = None
        view_default_11 = torch.ops.aten.view.default(clone_default_4, input_114);  clone_default_4 = None
        clone_default_5 = torch.ops.aten.clone.default(input_125, memory_format = torch.contiguous_format);  input_125 = None
        view_default_12 = torch.ops.aten.view.default(clone_default_5, input_116);  clone_default_5 = None
        bmm_default_2 = torch.ops.aten.bmm.default(view_default_11, view_default_12);  view_default_11 = view_default_12 = None
        view_default_13 = torch.ops.aten.view.default(bmm_default_2, input_117);  bmm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_13, add_tensor_1, alpha = 1);  view_default_13 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32)
        amax_default_2 = torch.ops.aten.amax.default(convert_element_type_default_6, [-1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_6, amax_default_2, alpha = 1);  convert_element_type_default_6 = None
        exp_default_2 = torch.ops.aten.exp.default(sub_tensor_2);  sub_tensor_2 = None
        clone_default_6 = torch.ops.aten.clone.default(input_126, memory_format = torch.contiguous_format);  input_126 = None
        view_default_14 = torch.ops.aten.view.default(clone_default_6, input_114);  clone_default_6 = None
        clone_default_7 = torch.ops.aten.clone.default(input_127, memory_format = torch.contiguous_format);  input_127 = None
        view_default_15 = torch.ops.aten.view.default(clone_default_7, input_116);  clone_default_7 = None
        bmm_default_3 = torch.ops.aten.bmm.default(view_default_14, view_default_15);  view_default_14 = view_default_15 = None
        view_default_16 = torch.ops.aten.view.default(bmm_default_3, input_117);  bmm_default_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_16, add_tensor_1, alpha = 1);  view_default_16 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(convert_element_type_default_7, torch.float32)
        amax_default_3 = torch.ops.aten.amax.default(convert_element_type_default_8, [-1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_8, amax_default_3, alpha = 1);  convert_element_type_default_8 = None
        exp_default_3 = torch.ops.aten.exp.default(sub_tensor_3);  sub_tensor_3 = None
        clone_default_8 = torch.ops.aten.clone.default(input_128, memory_format = torch.contiguous_format);  input_128 = None
        view_default_17 = torch.ops.aten.view.default(clone_default_8, input_114);  clone_default_8 = None
        clone_default_9 = torch.ops.aten.clone.default(input_129, memory_format = torch.contiguous_format);  input_129 = None
        view_default_18 = torch.ops.aten.view.default(clone_default_9, input_116);  clone_default_9 = None
        bmm_default_4 = torch.ops.aten.bmm.default(view_default_17, view_default_18);  view_default_17 = view_default_18 = None
        view_default_19 = torch.ops.aten.view.default(bmm_default_4, input_117);  bmm_default_4 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_19, add_tensor_1, alpha = 1);  view_default_19 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16);  add_tensor_6 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(convert_element_type_default_9, torch.float32)
        amax_default_4 = torch.ops.aten.amax.default(convert_element_type_default_10, [-1], True)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(convert_element_type_default_10, amax_default_4, alpha = 1);  convert_element_type_default_10 = None
        exp_default_4 = torch.ops.aten.exp.default(sub_tensor_4);  sub_tensor_4 = None
        clone_default_10 = torch.ops.aten.clone.default(input_130, memory_format = torch.contiguous_format);  input_130 = None
        view_default_20 = torch.ops.aten.view.default(clone_default_10, input_114);  clone_default_10 = input_114 = None
        clone_default_11 = torch.ops.aten.clone.default(input_131, memory_format = torch.contiguous_format);  input_131 = None
        view_default_21 = torch.ops.aten.view.default(clone_default_11, input_116);  clone_default_11 = input_116 = None
        bmm_default_5 = torch.ops.aten.bmm.default(view_default_20, view_default_21);  view_default_20 = view_default_21 = None
        view_default_22 = torch.ops.aten.view.default(bmm_default_5, input_117);  bmm_default_5 = input_117 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(view_default_22, add_tensor_1, alpha = 1);  view_default_22 = add_tensor_1 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(convert_element_type_default_11, torch.float32)
        amax_default_5 = torch.ops.aten.amax.default(convert_element_type_default_12, [-1], True)
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_12, amax_default_5, alpha = 1);  convert_element_type_default_12 = None
        exp_default_5 = torch.ops.aten.exp.default(sub_tensor_5);  sub_tensor_5 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1]);  view_default_5 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1]);  view_default_6 = None
        return (bmm_default, amax_default, sum_dim_int_list, convert_element_type_default_2, convert_element_type_default_3, amax_default_1, exp_default_1, convert_element_type_default_5, amax_default_2, exp_default_2, convert_element_type_default_7, amax_default_3, exp_default_3, convert_element_type_default_9, amax_default_4, exp_default_4, convert_element_type_default_11, amax_default_5, exp_default_5, permute_default_5, permute_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
