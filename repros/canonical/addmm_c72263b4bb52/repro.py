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
    def forward(self, input_15, input_16, input_17, input_18, input_19, input_20, input_21, input_22):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_15, torch.float32);  input_15 = None
        getitem = input_16[1];  input_16 = None
        add_tensor = torch.ops.aten.add.Tensor(input_17, 1e-06, alpha = 1);  input_17 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem, alpha = 1);  convert_element_type_default = getitem = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_18);  mul_tensor = input_18 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_19, alpha = 1);  mul_tensor_1 = input_19 = None
        permute_default = torch.ops.aten.permute.default(add_tensor_1, [0, 3, 1, 2]);  add_tensor_1 = None
        view_default = torch.ops.aten.view.default(permute_default, input_20);  permute_default = input_20 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_21, torch.bfloat16);  input_21 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_22, torch.bfloat16);  input_22 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_1, convert_element_type_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_1 = convert_element_type_default_3 = permute_default_1 = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
