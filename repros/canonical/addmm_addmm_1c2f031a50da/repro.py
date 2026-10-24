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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60):
        add_tensor = torch.ops.aten.add.Tensor(input_44, input_45, alpha = 1);  input_44 = input_45 = None
        select_int = torch.ops.aten.select.int(input_46, 3, 0);  input_46 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, select_int, alpha = 1);  add_tensor = select_int = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_47);  mul_tensor = input_47 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_48, alpha = 1);  mul_tensor_1 = input_48 = None
        view_default = torch.ops.aten.view.default(add_tensor_3, input_49);  input_49 = None
        permute_default = torch.ops.aten.permute.default(input_50, [1, 0]);  input_50 = None
        addmm_default = torch.ops.aten.addmm.default(input_51, view_default, permute_default, beta = 1, alpha = 1);  input_51 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_52);  addmm_default = input_52 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_1, 0.5)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(view_default_1, 3.0)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(pow_tensor_scalar, 0.044715);  pow_tensor_scalar = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_3, alpha = 1);  view_default_1 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_4, 0.7978845608028654);  add_tensor_4 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_4);  mul_tensor_4 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_5);  mul_tensor_2 = add_tensor_5 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_5, input_53);  mul_tensor_5 = input_53 = None
        permute_default_1 = torch.ops.aten.permute.default(input_54, [1, 0]);  input_54 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_55, view_default_2, permute_default_1, beta = 1, alpha = 1);  input_55 = view_default_2 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_56);  addmm_default_1 = input_56 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_3, add_tensor_3, alpha = 1);  view_default_3 = add_tensor_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_6, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_6, getitem_3, alpha = 1);  add_tensor_6 = getitem_3 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(getitem_2, 1e-12, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = rsqrt_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_57);  mul_tensor_6 = input_57 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_7, input_58, alpha = 1);  mul_tensor_7 = input_58 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.complex64)
        _fft_c2c_default = torch.ops.aten._fft_c2c.default(convert_element_type_default, [1, 2], 0, True);  convert_element_type_default = None
        select_int_1 = torch.ops.aten.select.int(input_59, 3, 0);  input_59 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_8, select_int_1, alpha = 1);  add_tensor_8 = select_int_1 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(add_tensor_9, [2], correction = 0, keepdim = True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_9, input_60, alpha = 1);  add_tensor_9 = input_60 = None
        return (_fft_c2c_default, var_mean_correction_2, sub_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
