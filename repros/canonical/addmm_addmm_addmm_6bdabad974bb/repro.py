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
    def forward(self, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93):
        full_default = torch.ops.aten.full.default(input_68, 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False);  input_68 = None
        add_tensor = torch.ops.aten.add.Tensor(input_69, input_70, alpha = 1);  input_69 = input_70 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_71, 1.1111111111111112);  input_71 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, add_tensor, alpha = 1);  mul_tensor = add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-07, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_72);  mul_tensor_1 = input_72 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_2, input_73, alpha = 1);  mul_tensor_2 = input_73 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_74, torch.bfloat16);  input_74 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_75, torch.bfloat16);  input_75 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_76);  convert_element_type_default_2 = input_76 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_77);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_78);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_79);  clone_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_80, torch.bfloat16);  input_80 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_81, torch.bfloat16);  input_81 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_77);  addmm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_78);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_79);  clone_default_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_82, torch.bfloat16);  input_82 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_83, torch.bfloat16);  input_83 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_6, [1, 0]);  convert_element_type_default_6 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_5 = view_default = permute_default_4 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_2, input_77);  addmm_default_2 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_78);  view_default_7 = input_78 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default_5, memory_format = torch.contiguous_format);  permute_default_5 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_2, input_79);  clone_default_2 = input_79 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1]);  view_default_6 = None
        div_tensor = torch.ops.aten.div.Tensor(permute_default_6, full_default);  permute_default_6 = full_default = None
        bmm_default = torch.ops.aten.bmm.default(view_default_3, div_tensor)
        view_default_10 = torch.ops.aten.view.default(bmm_default, input_84);  bmm_default = input_84 = None
        where_self = torch.ops.aten.where.self(input_85, input_86, view_default_10);  input_85 = input_86 = view_default_10 = None
        bmm_default_1 = torch.ops.aten.bmm.default(input_87, view_default_9);  input_87 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_88, input_89, input_90, beta = 1, alpha = 1);  input_88 = input_89 = input_90 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_3, input_77);  addmm_default_3 = input_77 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_91, torch.bfloat16);  input_91 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_7, 0.1);  convert_element_type_default_7 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(gt_scalar, view_default_11);  gt_scalar = view_default_11 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_4, add_tensor_3, alpha = 1);  mul_tensor_4 = add_tensor_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_4, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-07, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_4, getitem_3, alpha = 1);  add_tensor_4 = getitem_3 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_92);  input_92 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_6, input_93, alpha = 1);  mul_tensor_6 = input_93 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(rsqrt_default_1, 1536);  rsqrt_default_1 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_9, [0, 2, 1]);  view_default_9 = None
        permute_default_8 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
        permute_default_9 = torch.ops.aten.permute.default(div_tensor, [0, 2, 1]);  div_tensor = None
        return (where_self, bmm_default_1, mul_tensor_5, add_tensor_6, div_tensor_1, permute_default_7, permute_default_8, permute_default_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
