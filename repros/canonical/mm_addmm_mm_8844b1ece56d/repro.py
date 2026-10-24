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
    def forward(self, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85):
        clone_default = torch.ops.aten.clone.default(input_68, memory_format = torch.contiguous_format);  input_68 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(clone_default, torch.float32);  clone_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_69);  mul_tensor = input_69 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_70, alpha = 1);  mul_tensor_1 = input_70 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_71)
        permute_default = torch.ops.aten.permute.default(input_72, [1, 0]);  input_72 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_73);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_74);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_1, input_71)
        permute_default_2 = torch.ops.aten.permute.default(input_75, [1, 0]);  input_75 = None
        addmm_default = torch.ops.aten.addmm.default(input_76, view_default_3, permute_default_2, beta = 1, alpha = 1);  input_76 = view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default, input_73);  addmm_default = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_74);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_1, input_71)
        permute_default_4 = torch.ops.aten.permute.default(input_77, [1, 0]);  input_77 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_6, permute_default_4);  view_default_6 = permute_default_4 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_1, input_73);  mm_default_1 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_74);  view_default_7 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        clone_default_3 = torch.ops.aten.clone.default(permute_default_5, memory_format = torch.contiguous_format);  permute_default_5 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_1, input_71)
        permute_default_6 = torch.ops.aten.permute.default(input_78, [1, 0]);  input_78 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_79, view_default_9, permute_default_6, beta = 1, alpha = 1);  input_79 = view_default_9 = permute_default_6 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_1, input_73);  addmm_default_1 = None
        view_default_11 = torch.ops.aten.view.default(view_default_10, input_74);  view_default_10 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_11, [0, 2, 1, 3]);  view_default_11 = None
        clone_default_4 = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        view_default_12 = torch.ops.aten.view.default(convert_element_type_default_1, input_71)
        permute_default_8 = torch.ops.aten.permute.default(input_80, [1, 0]);  input_80 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_12, permute_default_8);  view_default_12 = permute_default_8 = None
        view_default_13 = torch.ops.aten.view.default(mm_default_2, input_73);  mm_default_2 = None
        view_default_14 = torch.ops.aten.view.default(view_default_13, input_74);  view_default_13 = None
        permute_default_9 = torch.ops.aten.permute.default(view_default_14, [0, 2, 1, 3]);  view_default_14 = None
        clone_default_5 = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        view_default_15 = torch.ops.aten.view.default(convert_element_type_default_1, input_71)
        permute_default_10 = torch.ops.aten.permute.default(input_81, [1, 0]);  input_81 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_82, view_default_15, permute_default_10, beta = 1, alpha = 1);  input_82 = view_default_15 = permute_default_10 = None
        view_default_16 = torch.ops.aten.view.default(addmm_default_2, input_73);  addmm_default_2 = None
        view_default_17 = torch.ops.aten.view.default(view_default_16, input_74);  view_default_16 = None
        permute_default_11 = torch.ops.aten.permute.default(view_default_17, [0, 2, 1, 3]);  view_default_17 = None
        clone_default_6 = torch.ops.aten.clone.default(permute_default_11, memory_format = torch.contiguous_format);  permute_default_11 = None
        view_default_18 = torch.ops.aten.view.default(convert_element_type_default_1, input_71)
        permute_default_12 = torch.ops.aten.permute.default(input_83, [1, 0]);  input_83 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_18, permute_default_12);  view_default_18 = permute_default_12 = None
        view_default_19 = torch.ops.aten.view.default(mm_default_3, input_73);  mm_default_3 = None
        view_default_20 = torch.ops.aten.view.default(view_default_19, input_74);  view_default_19 = None
        permute_default_13 = torch.ops.aten.permute.default(view_default_20, [0, 2, 1, 3]);  view_default_20 = None
        clone_default_7 = torch.ops.aten.clone.default(permute_default_13, memory_format = torch.contiguous_format);  permute_default_13 = None
        view_default_21 = torch.ops.aten.view.default(convert_element_type_default_1, input_71);  convert_element_type_default_1 = input_71 = None
        permute_default_14 = torch.ops.aten.permute.default(input_84, [1, 0]);  input_84 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_85, view_default_21, permute_default_14, beta = 1, alpha = 1);  input_85 = view_default_21 = permute_default_14 = None
        view_default_22 = torch.ops.aten.view.default(addmm_default_3, input_73);  addmm_default_3 = input_73 = None
        view_default_23 = torch.ops.aten.view.default(view_default_22, input_74);  view_default_22 = input_74 = None
        permute_default_15 = torch.ops.aten.permute.default(view_default_23, [0, 2, 1, 3]);  view_default_23 = None
        clone_default_8 = torch.ops.aten.clone.default(permute_default_15, memory_format = torch.contiguous_format);  permute_default_15 = None
        return (clone_default_1, clone_default_2, clone_default_3, clone_default_4, clone_default_5, clone_default_6, clone_default_7, clone_default_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
