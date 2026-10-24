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
    def forward(self, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63):
        rsqrt_default = torch.ops.aten.rsqrt.default(input_40);  input_40 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_41, input_42, alpha = 1);  input_41 = input_42 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_43);  mul_tensor = input_43 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, input_44, alpha = 1);  mul_tensor_1 = input_44 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_45);  convert_element_type_default = None
        addmm_default = torch.ops.aten.addmm.default(input_46, view_default, input_47, beta = 1, alpha = 1);  input_46 = input_47 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_48, view_default, input_49, beta = 1, alpha = 1);  input_48 = input_49 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_50, view_default, input_51, beta = 1, alpha = 1);  input_50 = view_default = input_51 = None
        clone_default = torch.ops.aten.clone.default(input_52, memory_format = torch.contiguous_format);  input_52 = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_53);  clone_default = input_53 = None
        view_default_2 = torch.ops.aten.view.default(input_54, input_55);  input_54 = input_55 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_2, torch.bfloat16);  view_default_2 = None
        bmm_default = torch.ops.aten.bmm.default(convert_element_type_default_1, view_default_1);  convert_element_type_default_1 = view_default_1 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default, input_56);  bmm_default = input_56 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_4 = torch.ops.aten.view.default(clone_default_1, input_57);  clone_default_1 = input_57 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_59, torch.bfloat16);  input_59 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_45);  view_default_4 = input_45 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_2, view_default_5, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_2 = permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_3, input_60);  addmm_default_3 = input_60 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_61, torch.bfloat16);  input_61 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_4, 0.1);  convert_element_type_default_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar, view_default_6);  gt_scalar = view_default_6 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_3, add_tensor, alpha = 1);  mul_tensor_3 = add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-07, alpha = 1);  getitem = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_62);  input_62 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_5, input_63, alpha = 1);  mul_tensor_5 = input_63 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_1, 1536);  rsqrt_default_1 = None
        return (addmm_default, addmm_default_1, addmm_default_2, view_default_5, mul_tensor_4, add_tensor_3, div_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
