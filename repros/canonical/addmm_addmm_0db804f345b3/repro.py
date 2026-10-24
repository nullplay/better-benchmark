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
    def forward(self, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27):
        add_tensor = torch.ops.aten.add.Tensor(input_20, input_21, alpha = 1);  input_20 = input_21 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-06, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_22);  mul_tensor = input_22 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_23, alpha = 1);  mul_tensor_1 = input_23 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        select_int = torch.ops.aten.select.int(convert_element_type_default_1, 1, 0)
        permute_default = torch.ops.aten.permute.default(input_24, [1, 0]);  input_24 = None
        addmm_default = torch.ops.aten.addmm.default(input_25, select_int, permute_default, beta = 1, alpha = 1);  input_25 = select_int = permute_default = None
        select_int_1 = torch.ops.aten.select.int(convert_element_type_default_1, 1, 1);  convert_element_type_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(input_26, [1, 0]);  input_26 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_27, select_int_1, permute_default_1, beta = 1, alpha = 1);  input_27 = select_int_1 = permute_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(addmm_default, addmm_default_1, alpha = 1);  addmm_default = addmm_default_1 = None
        div_tensor = torch.ops.aten.div.Tensor(add_tensor_3, 2);  add_tensor_3 = None
        return (div_tensor,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
