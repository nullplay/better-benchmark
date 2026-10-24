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
    def forward(self, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_19, torch.bfloat16);  input_19 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_20, torch.bfloat16);  input_20 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        view_default = torch.ops.aten.view.default(mul_tensor_1, input_21);  mul_tensor_1 = input_21 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_22);  input_22 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_23);  permute_default = input_23 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_24);  mm_default = input_24 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        full_default = torch.ops.aten.full.default(input_25, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_25 = None
        where_self = torch.ops.aten.where.self(input_26, full_default, view_default_1);  input_26 = full_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(where_self, input_27);  where_self = input_27 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_28);  input_28 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_29);  permute_default_1 = input_29 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True, dtype = torch.float32);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list, input_30);  sum_dim_int_list = input_30 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_31);  mm_default_2 = input_31 = None
        return (convert_element_type_default_2, mm_default_3, view_default_3, view_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
