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
    def forward(self, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38):
        relu_default = torch.ops.aten.relu.default(input_29);  input_29 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(relu_default, torch.float32);  relu_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_30);  mul_tensor = input_30 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_31, alpha = 1);  mul_tensor_1 = input_31 = None
        permute_default = torch.ops.aten.permute.default(input_32, [1, 0]);  input_32 = None
        cat_default = torch.ops.aten.cat.default([permute_default, input_33], 0);  permute_default = input_33 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(cat_default, torch.bfloat16);  cat_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_34);  convert_element_type_default_2 = input_34 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(convert_element_type_default_1, [0, 6, 0, 0], 0);  convert_element_type_default_1 = None
        mm_default = torch.ops.aten.mm.default(view_default, constant_pad_nd_default);  view_default = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -6, 1);  mm_default = None
        view_default_1 = torch.ops.aten.view.default(slice_tensor, input_35);  slice_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_1, input_36, alpha = 1);  view_default_1 = input_36 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_3, input_37);  convert_element_type_default_3 = input_37 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_35);  input_35 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_4, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, amax_default, alpha = 1);  convert_element_type_default_4 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, input_38, alpha = 1);  sub_tensor_1 = input_38 = None
        return (view_default_3, exp_default, sub_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
