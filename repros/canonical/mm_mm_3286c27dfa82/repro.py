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
    def forward(self, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30):
        mm_default = torch.ops.aten.mm.default(input_23, input_24);  input_24 = None
        permute_default = torch.ops.aten.permute.default(input_23, [1, 0]);  input_23 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_25);  permute_default = input_25 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_26);  convert_element_type_default = input_26 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default, [0, 2, 3, 1]);  view_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(permute_default_1, input_27);  input_27 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 640)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True, dtype = None)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_28, torch.float32);  input_28 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_2, input_29, alpha = 1);  convert_element_type_default_2 = input_29 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_30);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4, alpha = 1);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor = torch.ops.aten.div.Tensor(input_30, 640);  input_30 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(permute_default_1, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1, 2], False, dtype = None);  mul_tensor_6 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(permute_default_1, [0, 1, 2], False, dtype = None);  permute_default_1 = None
        return (convert_element_type_default_1, mul_tensor_5, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
