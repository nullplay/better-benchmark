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
    def forward(self, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106):
        rsqrt_default = torch.ops.aten.rsqrt.default(input_77);  input_77 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_78, input_79, alpha = 1);  input_78 = input_79 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_80);  mul_tensor = input_80 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, input_81, alpha = 1);  mul_tensor_1 = input_81 = None
        view_default = torch.ops.aten.view.default(input_82, input_83);  input_82 = input_83 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2, 4, 5]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_84);  clone_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_1, alpha = 1);  add_tensor = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_85);  add_tensor_1 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(view_default_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(view_default_2, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_86);  mul_tensor_2 = input_86 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_3, input_87, alpha = 1);  mul_tensor_3 = input_87 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_88, torch.bfloat16);  input_88 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_1, input_89);  convert_element_type_default_1 = input_89 = None
        permute_default_1 = torch.ops.aten.permute.default(input_90, [1, 0]);  input_90 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default = view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default, input_91);  input_91 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_4);  mul_tensor_4 = add_tensor_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_92, torch.bfloat16);  input_92 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_93, torch.bfloat16);  input_93 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_3, input_94);  convert_element_type_default_3 = input_94 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_5, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_95);  addmm_default_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_2, view_default_6, alpha = 1);  view_default_2 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_5, input_96);  add_tensor_5 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(view_default_7, [3], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(view_default_7, getitem_3, alpha = 1);  getitem_3 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_97);  input_97 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_8, input_98, alpha = 1);  mul_tensor_8 = input_98 = None
        index_tensor = torch.ops.aten.index.Tensor(add_tensor_7, [None, input_99]);  add_tensor_7 = input_99 = None
        view_default_8 = torch.ops.aten.view.default(input_100, input_84);  input_100 = input_84 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(input_101, 53, alpha = 1);  input_101 = None
        fmod_scalar = torch.ops.aten.fmod.Scalar(add_tensor_8, 56);  add_tensor_8 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(view_default_8, [None, fmod_scalar]);  view_default_8 = None
        index_tensor_2 = torch.ops.aten.index.Tensor(index_tensor_1, [None, None, fmod_scalar]);  index_tensor_1 = fmod_scalar = None
        lt_scalar = torch.ops.aten.lt.Scalar(input_102, 0.9956521736457944);  input_102 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(lt_scalar, torch.bfloat16);  lt_scalar = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_6, 0.9956521736457944);  convert_element_type_default_6 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(index_tensor_2, div_tensor);  index_tensor_2 = div_tensor = None
        add_tensor_9 = torch.ops.aten.add.Tensor(view_default_7, mul_tensor_9, alpha = 1);  view_default_7 = mul_tensor_9 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_9, input_85);  add_tensor_9 = input_85 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(view_default_9, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_10);  add_tensor_10 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(view_default_9, getitem_5, alpha = 1);  getitem_5 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(sub_tensor_3, rsqrt_default_3);  sub_tensor_3 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, input_103);  input_103 = None
        view_default_10 = torch.ops.aten.view.default(input_104, input_95);  input_104 = input_95 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(input_105, 0.9956521736457944);  input_105 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(view_default_10, div_tensor_1);  view_default_10 = div_tensor_1 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(view_default_9, mul_tensor_12, alpha = 1);  view_default_9 = mul_tensor_12 = None
        view_default_11 = torch.ops.aten.view.default(add_tensor_11, input_96);  add_tensor_11 = input_96 = None
        view_default_12 = torch.ops.aten.view.default(view_default_11, input_106);  view_default_11 = input_106 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(rsqrt_default_2, 128);  rsqrt_default_2 = None
        return (addmm_default, view_default_5, mul_tensor_7, index_tensor, rsqrt_default_3, mul_tensor_10, mul_tensor_11, view_default_12, div_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
