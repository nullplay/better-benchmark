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
    def forward(self, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40):
        view_default = torch.ops.aten.view.default(input_25, input_26);  input_25 = input_26 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_27);  view_default = input_27 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_28);  mm_default = input_28 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_29, torch.bfloat16);  input_29 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        full_default = torch.ops.aten.full.default(input_30, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_30 = None
        where_self = torch.ops.aten.where.self(input_31, full_default, mul_tensor_1);  input_31 = full_default = mul_tensor_1 = None
        view_default_2 = torch.ops.aten.view.default(where_self, input_32);  where_self = input_32 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_33);  permute_default = input_33 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_34);  view_default_2 = input_34 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_2, input_35);  mm_default_2 = input_35 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_36);  input_36 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_37, input_38)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, mul_tensor_3);  convert_element_type_default_2 = mul_tensor_3 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True, dtype = torch.float32);  mul_tensor_4 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_39);  sum_dim_int_list = input_39 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_37);  input_37 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_38);  mul_tensor_2 = input_38 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = torch.float32);  mul_tensor_5 = None
        add_tensor = torch.ops.aten.add.Tensor(input_40, mul_tensor_6, alpha = 1);  input_40 = mul_tensor_6 = None
        return (convert_element_type_default_3, view_default_4, sum_dim_int_list_1, add_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
