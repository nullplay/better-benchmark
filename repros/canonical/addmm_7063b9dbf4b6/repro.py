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
    def forward(self, input_13, input_14, input_15, input_16, input_17, input_18, input_19, input_20):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_13, torch.float32);  input_13 = None
        getitem = input_14[1];  input_14 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem, alpha = 1);  convert_element_type_default = getitem = None
        add_tensor = torch.ops.aten.add.Tensor(input_15, 1e-06, alpha = 1);  input_15 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_16);  mul_tensor = input_16 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_17, alpha = 1);  mul_tensor_1 = input_17 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [0, 3, 1, 2]);  convert_element_type_default_1 = None
        view_default = torch.ops.aten.view.default(permute_default, input_18);  permute_default = input_18 = None
        permute_default_1 = torch.ops.aten.permute.default(input_19, [1, 0]);  input_19 = None
        addmm_default = torch.ops.aten.addmm.default(input_20, view_default, permute_default_1, beta = 1, alpha = 1);  input_20 = view_default = permute_default_1 = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
