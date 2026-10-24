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
    def forward(self, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52):
        add_tensor = torch.ops.aten.add.Tensor(input_36, input_37, alpha = 1);  input_36 = input_37 = None
        view_default = torch.ops.aten.view.default(input_38, input_39);  input_38 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        addmm_default = torch.ops.aten.addmm.default(input_40, input_41, input_42, beta = 1, alpha = 1);  input_40 = input_41 = input_42 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_39);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        mean_dim = torch.ops.aten.mean.dim(add_tensor_2, [-1], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, mean_dim, alpha = 1);  mean_dim = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_43, sub_tensor);  input_43 = sub_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_correction = torch.ops.aten.var.correction(convert_element_type_default, [-1], correction = 1.0, keepdim = True);  convert_element_type_default = None
        sqrt_default = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(sqrt_default, torch.bfloat16);  sqrt_default = None
        add_tensor_3 = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-06, alpha = 1);  convert_element_type_default_1 = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_3);  mul_tensor = add_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(div_tensor, input_44, alpha = 1);  div_tensor = input_44 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_4, input_45);  add_tensor_4 = input_45 = None
        permute_default = torch.ops.aten.permute.default(input_46, [1, 0]);  input_46 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_47, view_default_2, permute_default, beta = 1, alpha = 1);  input_47 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_48);  addmm_default_1 = input_48 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_2);  mul_tensor_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_5);  mul_tensor_1 = add_tensor_5 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_3, input_49);  convert_element_type_default_3 = input_49 = None
        permute_default_1 = torch.ops.aten.permute.default(input_50, [1, 0]);  input_50 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_51, view_default_4, permute_default_1, beta = 1, alpha = 1);  input_51 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_39);  addmm_default_2 = input_39 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_5, alpha = 1);  add_tensor_2 = view_default_5 = None
        mean_dim_1 = torch.ops.aten.mean.dim(add_tensor_6, [-1], True, dtype = None)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_6, mean_dim_1, alpha = 1);  mean_dim_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, input_52, alpha = 1);  add_tensor_6 = input_52 = None
        return (sub_tensor_1, convert_element_type_default_4, add_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
