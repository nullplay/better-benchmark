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
    def forward(self, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93):
        view_default = torch.ops.aten.view.default(input_66, input_67);  input_66 = input_67 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [3], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_68);  mul_tensor = input_68 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_69, alpha = 1);  mul_tensor_1 = input_69 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_70);  convert_element_type_default_1 = input_70 = None
        permute_default = torch.ops.aten.permute.default(input_71, [1, 0]);  input_71 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default);  view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_72);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [3], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_3, alpha = 1);  convert_element_type_default_2 = getitem_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_73);  mul_tensor_2 = input_73 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_3, input_74, alpha = 1);  mul_tensor_3 = input_74 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_3, input_75);  convert_element_type_default_3 = input_75 = None
        view_default_4 = torch.ops.aten.view.default(input_76, input_77);  input_76 = input_77 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_78);  view_default_4 = input_78 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_79);  view_default_5 = input_79 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_6, [0, 1, 3, 2, 4, 5]);  view_default_6 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_80);  clone_default = input_80 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_2, view_default_7, alpha = 1);  view_default_2 = view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_4, input_81);  add_tensor_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_5, alpha = 1);  convert_element_type_default_4 = getitem_5 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = rsqrt_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_82);  mul_tensor_4 = input_82 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_5, input_83, alpha = 1);  mul_tensor_5 = input_83 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_84, torch.bfloat16);  input_84 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_5, input_85);  convert_element_type_default_5 = input_85 = None
        permute_default_2 = torch.ops.aten.permute.default(input_86, [1, 0]);  input_86 = None
        addmm_default = torch.ops.aten.addmm.default(input_87, view_default_9, permute_default_2, beta = 1, alpha = 1);  input_87 = view_default_9 = permute_default_2 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default, input_88);  addmm_default = input_88 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(view_default_8, view_default_10, alpha = 1);  view_default_8 = view_default_10 = None
        view_default_11 = torch.ops.aten.view.default(add_tensor_7, input_72);  add_tensor_7 = input_72 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_6, [3], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_6, getitem_7, alpha = 1);  convert_element_type_default_6 = getitem_7 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_89);  sub_tensor_3 = input_89 = None
        index_tensor = torch.ops.aten.index.Tensor(input_90, [None, input_91]);  input_90 = input_91 = None
        fmod_scalar = torch.ops.aten.fmod.Scalar(input_92, 28);  input_92 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar]);  index_tensor = fmod_scalar = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_11, index_tensor_1, alpha = 1);  view_default_11 = index_tensor_1 = None
        view_default_12 = torch.ops.aten.view.default(add_tensor_8, input_81);  add_tensor_8 = input_81 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_12, torch.float32)
        add_tensor_9 = torch.ops.aten.add.Tensor(view_default_12, input_93, alpha = 1);  view_default_12 = input_93 = None
        return (view_default_3, add_tensor_6, getitem_6, mul_tensor_6, convert_element_type_default_7, add_tensor_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
