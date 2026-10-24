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
    def forward(self, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        add_tensor = torch.ops.aten.add.Tensor(input_38, input_39, alpha = 1);  input_38 = input_39 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_40);  mul_tensor = input_40 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_41, alpha = 1);  mul_tensor_1 = input_41 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_42)
        permute_default = torch.ops.aten.permute.default(input_43, [1, 0]);  input_43 = None
        addmm_default = torch.ops.aten.addmm.default(input_44, view_default, permute_default, beta = 1, alpha = 1);  input_44 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_45);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_46);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        mul_scalar = torch.ops.aten.mul.Scalar(permute_default_1, 0.334370152488211);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_1, input_42)
        permute_default_2 = torch.ops.aten.permute.default(input_47, [1, 0]);  input_47 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_48, view_default_3, permute_default_2, beta = 1, alpha = 1);  input_48 = view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_45);  addmm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_46);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_3, [0, 1, 3, 2]);  permute_default_3 = None
        view_default_6 = torch.ops.aten.view.default(input_49, input_50);  input_49 = input_50 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_1, input_42);  convert_element_type_default_1 = input_42 = None
        permute_default_5 = torch.ops.aten.permute.default(input_51, [1, 0]);  input_51 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_52, view_default_7, permute_default_5, beta = 1, alpha = 1);  input_52 = view_default_7 = permute_default_5 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_2, input_45);  addmm_default_2 = input_45 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_46);  view_default_8 = input_46 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_9, [0, 2, 1, 3]);  view_default_9 = None
        expand_default = torch.ops.aten.expand.default(permute_default_6, input_53, implicit = False);  permute_default_6 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_10 = torch.ops.aten.view.default(clone_default, input_54);  clone_default = input_54 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_6, view_default_10);  view_default_6 = view_default_10 = None
        view_default_11 = torch.ops.aten.view.default(bmm_default, input_53);  bmm_default = input_53 = None
        return (mul_scalar, permute_default_4, view_default_11)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
