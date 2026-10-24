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
    def forward(self, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_21, torch.float32);  input_21 = None
        permute_default = torch.ops.aten.permute.default(input_22, [1, 0]);  input_22 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(permute_default, torch.float32);  permute_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1);  mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_23, torch.float32);  input_23 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2, alpha = 1);  mul_tensor_1 = mul_tensor_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        tanh_default = torch.ops.aten.tanh.default(convert_element_type_default_3);  convert_element_type_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(input_24, [1, 0]);  input_24 = None
        addmm_default = torch.ops.aten.addmm.default(input_25, tanh_default, permute_default_1, beta = 1, alpha = 1);  input_25 = tanh_default = permute_default_1 = None
        tanh_default_1 = torch.ops.aten.tanh.default(addmm_default);  addmm_default = None
        permute_default_2 = torch.ops.aten.permute.default(input_26, [1, 0]);  input_26 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_27, tanh_default_1, permute_default_2, beta = 1, alpha = 1);  input_27 = tanh_default_1 = permute_default_2 = None
        tanh_default_2 = torch.ops.aten.tanh.default(addmm_default_1);  addmm_default_1 = None
        permute_default_3 = torch.ops.aten.permute.default(input_28, [1, 0]);  input_28 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_29, tanh_default_2, permute_default_3, beta = 1, alpha = 1);  input_29 = tanh_default_2 = permute_default_3 = None
        tanh_default_3 = torch.ops.aten.tanh.default(addmm_default_2);  addmm_default_2 = None
        permute_default_4 = torch.ops.aten.permute.default(input_30, [1, 0]);  input_30 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_31, tanh_default_3, permute_default_4, beta = 1, alpha = 1);  input_31 = tanh_default_3 = permute_default_4 = None
        return (addmm_default_3,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
