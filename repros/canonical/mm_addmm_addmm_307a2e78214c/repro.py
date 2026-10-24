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
    def forward(self, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96):
        permute_default = torch.ops.aten.permute.default(input_70, [1, 0]);  input_70 = None
        view_default = torch.ops.aten.view.default(input_71, input_72);  input_71 = input_72 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_73)
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [3], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_74, alpha = 1);  convert_element_type_default = input_74 = None
        clone_default = torch.ops.aten.clone.default(input_75, memory_format = torch.contiguous_format);  input_75 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_76);  clone_default = input_76 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_77, torch.bfloat16);  input_77 = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_1, 0.9826086945831776);  convert_element_type_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_2, div_tensor);  view_default_2 = div_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_1, mul_tensor, alpha = 1);  view_default_1 = mul_tensor = None
        view_default_3 = torch.ops.aten.view.default(add_tensor, input_78);  add_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction_1[0]
        getitem_1 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1, alpha = 1);  convert_element_type_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_79);  mul_tensor_1 = input_79 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_80, torch.bfloat16);  input_80 = None
        view_default_4 = torch.ops.aten.view.default(input_81, input_82);  input_81 = input_82 = None
        permute_default_1 = torch.ops.aten.permute.default(input_83, [1, 0]);  input_83 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_4, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default, input_84);  input_84 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.5)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 0.7071067811865476);  convert_element_type_default_4 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_2);  mul_tensor_3 = add_tensor_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_85, torch.bfloat16);  input_85 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_86, torch.bfloat16);  input_86 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_5, input_87);  convert_element_type_default_5 = input_87 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_6, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_1, input_88);  addmm_default_1 = input_88 = None
        lt_scalar = torch.ops.aten.lt.Scalar(input_89, 0.9826086945831776);  input_89 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(lt_scalar, torch.bfloat16);  lt_scalar = None
        div_tensor_1 = torch.ops.aten.div.Tensor(convert_element_type_default_8, 0.9826086945831776);  convert_element_type_default_8 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(view_default_7, div_tensor_1);  view_default_7 = div_tensor_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_3, mul_tensor_6, alpha = 1);  mul_tensor_6 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_3, input_73);  add_tensor_3 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_9, [3], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_2[0]
        getitem_3 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_9, getitem_3, alpha = 1);  convert_element_type_default_9 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_1);  sub_tensor_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_90);  mul_tensor_7 = input_90 = None
        fmod_scalar = torch.ops.aten.fmod.Scalar(input_91, 14);  input_91 = None
        index_tensor = torch.ops.aten.index.Tensor(input_92, [None, fmod_scalar]);  input_92 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar]);  index_tensor = fmod_scalar = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_93, torch.bfloat16);  input_93 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(convert_element_type_default_10, 0.9782608672976494);  convert_element_type_default_10 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(index_tensor_1, div_tensor_2);  index_tensor_1 = div_tensor_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_8, mul_tensor_9, alpha = 1);  mul_tensor_9 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_5, input_78);  add_tensor_5 = input_78 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_11, [2], correction = 0, keepdim = True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_11, input_94, alpha = 1);  convert_element_type_default_11 = input_94 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_95, input_96);  input_95 = input_96 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_9, mul_tensor_10, alpha = 1);  mul_tensor_10 = None
        view_default_10 = torch.ops.aten.view.default(add_tensor_6, input_73);  add_tensor_6 = input_73 = None
        return (mm_default, var_mean_correction, sub_tensor, view_default_3, getitem_1, rsqrt_default, mul_tensor_2, addmm_default, view_default_6, view_default_8, getitem_3, rsqrt_default_1, mul_tensor_8, view_default_9, var_mean_correction_3, sub_tensor_3, view_default_10)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
