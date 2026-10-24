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
    def forward(self, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35):
        add_tensor = torch.ops.aten.add.Tensor(input_26, input_27, alpha = 1);  input_26 = input_27 = None
        view_default = torch.ops.aten.view.default(input_28, input_29);  input_28 = input_29 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-06, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_30);  mul_tensor = input_30 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_31, alpha = 1);  mul_tensor_1 = input_31 = None
        select_int = torch.ops.aten.select.int(add_tensor_3, 1, 0)
        select_int_1 = torch.ops.aten.select.int(add_tensor_3, 1, 1);  add_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_32, torch.bfloat16);  input_32 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(select_int, torch.bfloat16);  select_int = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, convert_element_type_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default = convert_element_type_default_2 = permute_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_34, torch.bfloat16);  input_34 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_35, torch.bfloat16);  input_35 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(select_int_1, torch.bfloat16);  select_int_1 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, convert_element_type_default_5, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = convert_element_type_default_5 = permute_default_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(addmm_default, addmm_default_1, alpha = 1);  addmm_default = addmm_default_1 = None
        div_tensor = torch.ops.aten.div.Tensor(add_tensor_4, 2);  add_tensor_4 = None
        return (div_tensor,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
