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
    def forward(self, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_32, torch.bfloat16);  input_32 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_34, torch.bfloat16);  input_34 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(permute_default, torch.float32);  permute_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_4, convert_element_type_default_5);  convert_element_type_default_4 = convert_element_type_default_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1);  mul_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1);  convert_element_type_default_3 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2, alpha = 1);  mul_tensor_1 = mul_tensor_2 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        tanh_default = torch.ops.aten.tanh.default(convert_element_type_default_6);  convert_element_type_default_6 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_35, torch.bfloat16);  input_35 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_36, torch.bfloat16);  input_36 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_8, [1, 0]);  convert_element_type_default_8 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_7, tanh_default, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_7 = tanh_default = permute_default_1 = None
        tanh_default_1 = torch.ops.aten.tanh.default(addmm_default);  addmm_default = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_37, torch.bfloat16);  input_37 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0]);  convert_element_type_default_10 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_9, tanh_default_1, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_9 = permute_default_2 = None
        tanh_default_2 = torch.ops.aten.tanh.default(addmm_default_1);  addmm_default_1 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(input_39, torch.bfloat16);  input_39 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(input_40, torch.bfloat16);  input_40 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_12, [1, 0]);  convert_element_type_default_12 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_11, tanh_default_2, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_11 = permute_default_3 = None
        tanh_default_3 = torch.ops.aten.tanh.default(addmm_default_2);  addmm_default_2 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        convert_element_type_default_14 = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_14, [1, 0]);  convert_element_type_default_14 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_13, tanh_default_3, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_13 = permute_default_4 = None
        return (tanh_default_1, tanh_default_2, tanh_default_3, addmm_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
