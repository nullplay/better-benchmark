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
    def forward(self, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78):
        add_tensor = torch.ops.aten.add.Tensor(input_55, input_56, alpha = 1);  input_55 = input_56 = None
        view_default = torch.ops.aten.view.default(add_tensor, input_57);  add_tensor = None
        addmm_default = torch.ops.aten.addmm.default(input_58, input_59, input_60, beta = 1, alpha = 1);  input_58 = input_59 = input_60 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_61);  addmm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default, view_default_1, alpha = 1);  view_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_62);  add_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [3], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, input_63);  sub_tensor = input_63 = None
        clone_default = torch.ops.aten.clone.default(input_64, memory_format = torch.contiguous_format);  input_64 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_65);  clone_default = input_65 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_66, 25, alpha = 1);  input_66 = None
        fmod_scalar = torch.ops.aten.fmod.Scalar(add_tensor_2, 28);  add_tensor_2 = None
        index_tensor = torch.ops.aten.index.Tensor(view_default_3, [None, fmod_scalar]);  view_default_3 = fmod_scalar = None
        iota_default = torch.ops.prims.iota.default(28, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor_3 = torch.ops.aten.add.Tensor(iota_default, 25, alpha = 1);  iota_default = None
        fmod_scalar_1 = torch.ops.aten.fmod.Scalar(add_tensor_3, 28);  add_tensor_3 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar_1]);  index_tensor = fmod_scalar_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_2, index_tensor_1, alpha = 1);  view_default_2 = index_tensor_1 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_4, input_57);  add_tensor_4 = input_57 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_3, alpha = 1);  convert_element_type_default_1 = getitem_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_67);  mul_tensor_1 = input_67 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_2, input_68, alpha = 1);  mul_tensor_2 = input_68 = None
        view_default_5 = torch.ops.aten.view.default(input_69, input_70);  input_69 = input_70 = None
        permute_default = torch.ops.aten.permute.default(input_71, [1, 0]);  input_71 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_72, view_default_5, permute_default, beta = 1, alpha = 1);  input_72 = view_default_5 = permute_default = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_73);  addmm_default_1 = input_73 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_7);  mul_tensor_3 = add_tensor_7 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_3, input_74);  convert_element_type_default_3 = input_74 = None
        permute_default_1 = torch.ops.aten.permute.default(input_75, [1, 0]);  input_75 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_76, view_default_7, permute_default_1, beta = 1, alpha = 1);  input_76 = view_default_7 = permute_default_1 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_2, input_61);  addmm_default_2 = input_61 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_4, view_default_8, alpha = 1);  view_default_4 = view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_8, input_62);  add_tensor_8 = input_62 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_77);  view_default_9 = input_77 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_10, [0, 1, 3, 4, 2, 5]);  view_default_10 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_11 = torch.ops.aten.view.default(clone_default_1, input_78);  clone_default_1 = input_78 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32);  view_default_11 = None
        return (getitem, mul_tensor, add_tensor_6, convert_element_type_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
