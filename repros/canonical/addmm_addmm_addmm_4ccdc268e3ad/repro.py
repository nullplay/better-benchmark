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
    def forward(self, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108):
        add_tensor = torch.ops.aten.add.Tensor(input_78, input_79, alpha = 1);  input_78 = input_79 = None
        view_default = torch.ops.aten.view.default(add_tensor, input_80);  add_tensor = None
        addmm_default = torch.ops.aten.addmm.default(input_81, input_82, input_83, beta = 1, alpha = 1);  input_81 = input_82 = input_83 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_84);  addmm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default, view_default_1, alpha = 1);  view_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_85);  add_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [3], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, input_86);  sub_tensor = input_86 = None
        clone_default = torch.ops.aten.clone.default(input_87, memory_format = torch.contiguous_format);  input_87 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_88);  clone_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_89, 11, alpha = 1);  input_89 = None
        fmod_scalar = torch.ops.aten.fmod.Scalar(add_tensor_2, 14);  add_tensor_2 = None
        index_tensor = torch.ops.aten.index.Tensor(view_default_3, [None, fmod_scalar]);  view_default_3 = fmod_scalar = None
        iota_default = torch.ops.prims.iota.default(14, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_3 = torch.ops.aten.add.Tensor(iota_default, 11, alpha = 1);  iota_default = None
        fmod_scalar_1 = torch.ops.aten.fmod.Scalar(add_tensor_3, 14);  add_tensor_3 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar_1]);  index_tensor = fmod_scalar_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_2, index_tensor_1, alpha = 1);  view_default_2 = index_tensor_1 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_4, input_80);  add_tensor_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_3, alpha = 1);  convert_element_type_default_1 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_90);  mul_tensor_1 = input_90 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_2, input_91, alpha = 1);  mul_tensor_2 = input_91 = None
        view_default_5 = torch.ops.aten.view.default(input_92, input_93);  input_92 = input_93 = None
        permute_default = torch.ops.aten.permute.default(input_94, [1, 0]);  input_94 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_95, view_default_5, permute_default, beta = 1, alpha = 1);  input_95 = view_default_5 = permute_default = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_96);  addmm_default_1 = input_96 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_7);  mul_tensor_3 = add_tensor_7 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_3, input_97);  convert_element_type_default_3 = input_97 = None
        permute_default_1 = torch.ops.aten.permute.default(input_98, [1, 0]);  input_98 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_99, view_default_7, permute_default_1, beta = 1, alpha = 1);  input_99 = view_default_7 = permute_default_1 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_2, input_84);  addmm_default_2 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_4, view_default_8, alpha = 1);  view_default_4 = view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_8, input_85);  add_tensor_8 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [3], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_5, alpha = 1);  convert_element_type_default_4 = getitem_5 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_1);  sub_tensor_2 = rsqrt_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_100);  mul_tensor_6 = input_100 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor_7, input_101, alpha = 1);  mul_tensor_7 = input_101 = None
        view_default_10 = torch.ops.aten.view.default(input_102, input_103);  input_102 = input_103 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_10, [0, 1, 3, 2, 4, 5]);  view_default_10 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_11 = torch.ops.aten.view.default(clone_default_1, input_88);  clone_default_1 = input_88 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(view_default_9, view_default_11, alpha = 1);  view_default_9 = view_default_11 = None
        view_default_12 = torch.ops.aten.view.default(add_tensor_11, input_80);  add_tensor_11 = input_80 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_12, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [2], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, getitem_7, alpha = 1);  convert_element_type_default_5 = getitem_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_104);  sub_tensor_3 = input_104 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_105, input_106, input_107, beta = 1, alpha = 1);  input_105 = input_106 = input_107 = None
        view_default_13 = torch.ops.aten.view.default(addmm_default_3, input_84);  addmm_default_3 = input_84 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(view_default_12, view_default_13, alpha = 1);  view_default_12 = view_default_13 = None
        view_default_14 = torch.ops.aten.view.default(add_tensor_12, input_85);  add_tensor_12 = input_85 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_14, torch.float32)
        add_tensor_13 = torch.ops.aten.add.Tensor(view_default_14, input_108, alpha = 1);  view_default_14 = input_108 = None
        return (getitem, mul_tensor, add_tensor_6, add_tensor_10, getitem_6, mul_tensor_8, convert_element_type_default_6, add_tensor_13)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
