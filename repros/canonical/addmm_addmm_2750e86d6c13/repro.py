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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47):
        mul_tensor = torch.ops.aten.mul.Tensor(input_35, 1.1111111111111112);  input_35 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_36, 1.1111111111111112);  input_36 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor, alpha = 1);  mul_tensor_1 = mul_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, getitem_1, alpha = 1);  add_tensor = getitem_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_37);  mul_tensor_2 = input_37 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_3, input_38, alpha = 1);  mul_tensor_3 = input_38 = None
        view_default = torch.ops.aten.view.default(add_tensor_2, input_39);  add_tensor_2 = input_39 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_40, torch.bfloat16);  input_40 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_2, convert_element_type_default_1, convert_element_type_default, beta = 1, alpha = 1);  convert_element_type_default_2 = convert_element_type_default_1 = convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_42);  input_42 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default_1, 0.5)
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_3, 3.0);  convert_element_type_default_3 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_5, alpha = 1);  view_default_1 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_3, 0.7978845608028654);  add_tensor_3 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_6);  mul_tensor_6 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_4);  mul_tensor_4 = add_tensor_4 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_7, input_43);  mul_tensor_7 = input_43 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_44, torch.bfloat16);  input_44 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_2, torch.bfloat16);  view_default_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_45, torch.bfloat16);  input_45 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_6, convert_element_type_default_5, convert_element_type_default_4, beta = 1, alpha = 1);  convert_element_type_default_6 = convert_element_type_default_4 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_46);  addmm_default_1 = input_46 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_47, view_default_3);  input_47 = view_default_3 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        return (addmm_default, mul_tensor_8, permute_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
