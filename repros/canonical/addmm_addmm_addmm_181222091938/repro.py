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
    def forward(self, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85):
        mul_tensor = torch.ops.aten.mul.Tensor(input_59, input_60);  input_59 = input_60 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_61);  mul_tensor = input_61 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, input_62, alpha = 1);  mul_tensor_1 = input_62 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default = torch.ops.aten.view.default(input_63, input_64);  input_63 = input_64 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2, 4, 5]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_65);  clone_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convert_element_type_default, view_default_1, alpha = 1);  convert_element_type_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_66);  add_tensor_1 = None
        view_default_3 = torch.ops.aten.view.default(input_67, input_68);  input_67 = input_68 = None
        permute_default_1 = torch.ops.aten.permute.default(input_69, [1, 0]);  input_69 = None
        addmm_default = torch.ops.aten.addmm.default(input_70, view_default_3, permute_default_1, beta = 1, alpha = 1);  input_70 = view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default, input_71);  addmm_default = input_71 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.5)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.7071067811865476);  convert_element_type_default_1 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_2, input_72);  convert_element_type_default_2 = input_72 = None
        permute_default_2 = torch.ops.aten.permute.default(input_73, [1, 0]);  input_73 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_74, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_74 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_75);  addmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_2, view_default_6, alpha = 1);  view_default_2 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_3, input_76);  add_tensor_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [3], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_1, alpha = 1);  convert_element_type_default_3 = getitem_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_77);  mul_tensor_5 = input_77 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_6, input_78, alpha = 1);  mul_tensor_6 = input_78 = None
        clone_default_1 = torch.ops.aten.clone.default(input_79, memory_format = torch.contiguous_format);  input_79 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_1, input_65);  clone_default_1 = input_65 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(input_80, 53, alpha = 1);  input_80 = None
        fmod_scalar = torch.ops.aten.fmod.Scalar(add_tensor_6, 56);  add_tensor_6 = None
        index_tensor = torch.ops.aten.index.Tensor(view_default_8, [None, fmod_scalar]);  view_default_8 = fmod_scalar = None
        iota_default = torch.ops.prims.iota.default(56, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_7 = torch.ops.aten.add.Tensor(iota_default, 53, alpha = 1);  iota_default = None
        fmod_scalar_1 = torch.ops.aten.fmod.Scalar(add_tensor_7, 56);  add_tensor_7 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar_1]);  index_tensor = fmod_scalar_1 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_7, index_tensor_1, alpha = 1);  view_default_7 = index_tensor_1 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_8, input_66);  add_tensor_8 = input_66 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_3, alpha = 1);  convert_element_type_default_4 = getitem_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_1, input_81);  sub_tensor_1 = input_81 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_82, input_83, input_84, beta = 1, alpha = 1);  input_82 = input_83 = input_84 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_2, input_75);  addmm_default_2 = input_75 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(view_default_9, view_default_10, alpha = 1);  view_default_9 = view_default_10 = None
        view_default_11 = torch.ops.aten.view.default(add_tensor_9, input_76);  add_tensor_9 = input_76 = None
        view_default_12 = torch.ops.aten.view.default(view_default_11, input_85);  view_default_11 = input_85 = None
        return (add_tensor_5, getitem_2, mul_tensor_7, view_default_12)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
