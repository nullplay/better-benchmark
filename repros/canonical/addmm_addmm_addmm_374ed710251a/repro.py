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
    def forward(self, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47):
        getitem = input_34[0];  input_34 = None
        permute_default = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default = torch.ops.aten.view.default(permute_default, input_35);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_36);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(input_37, [1, 0]);  input_37 = None
        addmm_default = torch.ops.aten.addmm.default(input_38, view_default_1, permute_default_1, beta = 1, alpha = 1);  input_38 = view_default_1 = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_35);  addmm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem_1 = var_mean_correction[0]
        getitem_2 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_2, alpha = 1);  convert_element_type_default = getitem_2 = None
        add_tensor = torch.ops.aten.add.Tensor(getitem_1, 1e-06, alpha = 1);  getitem_1 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_39);  mul_tensor = input_39 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_40, alpha = 1);  mul_tensor_1 = input_40 = None
        view_default_3 = torch.ops.aten.view.default(input_41, input_36);  input_41 = input_36 = None
        permute_default_2 = torch.ops.aten.permute.default(input_42, [1, 0]);  input_42 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_43, view_default_3, permute_default_2, beta = 1, alpha = 1);  input_43 = view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_44);  addmm_default_1 = input_44 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.5)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 0.7071067811865476);  convert_element_type_default_1 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_2, input_45);  convert_element_type_default_2 = input_45 = None
        permute_default_3 = torch.ops.aten.permute.default(input_46, [1, 0]);  input_46 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_47, view_default_5, permute_default_3, beta = 1, alpha = 1);  input_47 = view_default_5 = permute_default_3 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_35);  addmm_default_2 = input_35 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_2, view_default_6, alpha = 1);  view_default_2 = view_default_6 = None
        select_int = torch.ops.aten.select.int(add_tensor_3, 1, 0);  add_tensor_3 = None
        return (add_tensor_1, select_int)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
