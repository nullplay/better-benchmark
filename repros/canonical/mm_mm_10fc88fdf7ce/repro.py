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
    def forward(self, input_15, input_16, input_17, input_18, input_19, input_20, input_21):
        mm_default = torch.ops.aten.mm.default(input_15, input_16);  input_16 = None
        permute_default = torch.ops.aten.permute.default(input_15, [1, 0]);  input_15 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_17);  permute_default = input_17 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_18);  convert_element_type_default = input_18 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default, [0, 2, 3, 1]);  view_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(permute_default_1, input_19);  input_19 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 640)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_20, input_21);  input_20 = input_21 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(permute_default_1, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1, 2], False, dtype = None);  mul_tensor_4 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(permute_default_1, [0, 1, 2], False, dtype = None);  permute_default_1 = None
        return (convert_element_type_default_1, mul_tensor_1, sum_dim_int_list, mul_tensor_3, sum_dim_int_list_1, sum_dim_int_list_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
