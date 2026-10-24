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
    def forward(self, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44):
        add_tensor = torch.ops.aten.add.Tensor(input_32, input_33, alpha = 1);  input_32 = input_33 = None
        view_default = torch.ops.aten.view.default(input_34, input_35);  input_34 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-06, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_36);  mul_tensor = input_36 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_37, alpha = 1);  mul_tensor_1 = input_37 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_39, torch.bfloat16);  input_39 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_2, input_40);  convert_element_type_default_2 = input_40 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_1, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_41);  input_41 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 0.5)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 0.7071067811865476);  convert_element_type_default_3 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_4);  mul_tensor_2 = add_tensor_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_4, input_43);  convert_element_type_default_4 = input_43 = None
        permute_default_1 = torch.ops.aten.permute.default(input_44, [1, 0]);  input_44 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_5 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_35);  addmm_default_1 = input_35 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_4, alpha = 1);  add_tensor_1 = view_default_4 = None
        return (addmm_default, view_default_3, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
