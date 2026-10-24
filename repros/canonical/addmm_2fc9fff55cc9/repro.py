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
    def forward(self, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38):
        add_tensor = torch.ops.aten.add.Tensor(input_26, input_27, alpha = 1);  input_26 = input_27 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_28, input_29);  input_28 = input_29 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-06, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_30);  mul_tensor_1 = input_30 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_2, input_31, alpha = 1);  mul_tensor_2 = input_31 = None
        cat_default = torch.ops.aten.cat.default([input_32, input_33, input_34], 0);  input_32 = input_33 = input_34 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(cat_default, torch.bfloat16);  cat_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_35, torch.bfloat16);  input_35 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_36);  convert_element_type_default_2 = input_36 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_37);  addmm_default = input_37 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_38);  view_default_1 = input_38 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [2, 0, 3, 1, 4]);  view_default_2 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default_1, 0);  permute_default_1 = None
        getitem_2 = unbind_int[0]
        getitem_3 = unbind_int[1]
        getitem_4 = unbind_int[2];  unbind_int = None
        return (getitem_2, getitem_3, getitem_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
