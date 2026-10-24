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
    def forward(self, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35):
        view_default = torch.ops.aten.view.default(input_27, input_28);  input_27 = input_28 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_29);  mul_tensor = input_29 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_30, alpha = 1);  mul_tensor_1 = input_30 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_31, torch.bfloat16);  input_31 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_32, torch.bfloat16);  input_32 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_3, input_33);  convert_element_type_default_3 = input_33 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_1, permute_default, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_34);  addmm_default = input_34 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_35);  view_default_2 = input_35 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [2, 0, 3, 1, 4]);  view_default_3 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default_1, 0);  permute_default_1 = None
        getitem_2 = unbind_int[0]
        getitem_3 = unbind_int[1]
        getitem_4 = unbind_int[2];  unbind_int = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(getitem_2, [0, 4], 0.0);  getitem_2 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(getitem_3, [0, 4], 0.0);  getitem_3 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(getitem_4, [0, 4], 0.0);  getitem_4 = None
        return (constant_pad_nd_default, constant_pad_nd_default_1, constant_pad_nd_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
