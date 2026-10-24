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
    def forward(self, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86):
        rsqrt_default = torch.ops.aten.rsqrt.default(input_61);  input_61 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_62, input_63, alpha = 1);  input_62 = input_63 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_64);  mul_tensor = input_64 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, input_65, alpha = 1);  mul_tensor_1 = input_65 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_66, 0.1);  input_66 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor);  gt_scalar = add_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_67, torch.bfloat16);  input_67 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_68, torch.bfloat16);  input_68 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_69);  convert_element_type_default_2 = input_69 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_70);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_71);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_72, torch.bfloat16);  input_72 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_73, torch.bfloat16);  input_73 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_70);  addmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_71);  view_default_3 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_74, torch.bfloat16);  input_74 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_75, torch.bfloat16);  input_75 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_6, [1, 0]);  convert_element_type_default_6 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_5 = view_default = permute_default_4 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_70);  addmm_default_2 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_71);  view_default_5 = input_71 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        permute_default_6 = torch.ops.aten.permute.default(permute_default_3, [0, 1, 3, 2]);  permute_default_3 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_76, implicit = False);  permute_default_1 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_77);  clone_default = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_6, input_78, implicit = False);  permute_default_6 = input_78 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_8 = torch.ops.aten.view.default(input_79, input_80);  input_79 = input_80 = None
        expand_default_2 = torch.ops.aten.expand.default(permute_default_5, input_76, implicit = False);  permute_default_5 = None
        clone_default_2 = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_2, input_77);  clone_default_2 = input_77 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_8, view_default_9);  view_default_8 = None
        view_default_10 = torch.ops.aten.view.default(bmm_default, input_76);  bmm_default = input_76 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_81, input_82, input_83, beta = 1, alpha = 1);  input_81 = input_82 = input_83 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_3, input_70);  addmm_default_3 = input_70 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_84, torch.bfloat16);  input_84 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.1);  convert_element_type_default_7 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(gt_scalar_1, view_default_11);  gt_scalar_1 = view_default_11 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_5, mul_tensor_3, alpha = 1);  mul_tensor_5 = mul_tensor_3 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_85);  input_85 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_7, input_86, alpha = 1);  mul_tensor_7 = input_86 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_1, 768);  rsqrt_default_1 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_9, [0, 2, 1]);  view_default_9 = None
        return (view_default_7, clone_default_1, view_default_10, mul_tensor_6, add_tensor_3, div_tensor, permute_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
