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
        add_tensor = torch.ops.aten.add.Tensor(input_15, input_16, alpha = 1);  input_15 = input_16 = None
        getitem = input_17[1];  input_17 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_18, 1e-06, alpha = 1);  input_18 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, getitem, alpha = 1);  add_tensor = getitem = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_19);  mul_tensor = input_19 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_20, alpha = 1);  mul_tensor_1 = input_20 = None
        select_int = torch.ops.aten.select.int(add_tensor_2, 1, 0);  add_tensor_2 = None
        clone_default = torch.ops.aten.clone.default(select_int, memory_format = None);  select_int = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_21, torch.bfloat16);  input_21 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_22, torch.bfloat16);  input_22 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(clone_default, torch.bfloat16);  clone_default = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, convert_element_type_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default = convert_element_type_default_2 = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
