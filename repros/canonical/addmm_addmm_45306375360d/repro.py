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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62):
        view_default = torch.ops.aten.view.default(input_44, input_45);  input_44 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_46, input_47);  input_46 = input_47 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, mul_tensor, alpha = 1);  view_default = mul_tensor = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_48);  add_tensor = None
        permute_default = torch.ops.aten.permute.default(input_49, [0, 1, 3, 2, 4, 5]);  input_49 = None
        view_default_2 = torch.ops.aten.view.default(permute_default, input_50);  permute_default = input_50 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8999999985098839);  convert_element_type_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default_2, div_tensor);  view_default_2 = div_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_1, alpha = 1);  view_default_1 = mul_tensor_1 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_1, input_45);  add_tensor_1 = input_45 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_52, torch.bfloat16);  input_52 = None
        view_default_4 = torch.ops.aten.view.default(input_53, input_54);  input_53 = input_54 = None
        permute_default_1 = torch.ops.aten.permute.default(input_55, [1, 0]);  input_55 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_4, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default, input_56);  input_56 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_57, torch.bfloat16);  input_57 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_3, input_59);  convert_element_type_default_3 = input_59 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_6, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_1, input_60);  addmm_default_1 = input_60 = None
        lt_scalar = torch.ops.aten.lt.Scalar(input_61, 0.8999999985098839);  input_61 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(lt_scalar, torch.bfloat16);  lt_scalar = None
        div_tensor_1 = torch.ops.aten.div.Tensor(convert_element_type_default_6, 0.8999999985098839);  convert_element_type_default_6 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_7, div_tensor_1);  view_default_7 = div_tensor_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_3, mul_tensor_5, alpha = 1);  view_default_3 = mul_tensor_5 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_3, input_48);  add_tensor_3 = input_48 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_7, [3], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_4 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_7, getitem_1, alpha = 1);  convert_element_type_default_7 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_62);  mul_tensor_6 = input_62 = None
        return (addmm_default, view_default_6, view_default_8, getitem_1, rsqrt_default, mul_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
