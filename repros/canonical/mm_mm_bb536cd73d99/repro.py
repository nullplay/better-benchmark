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
    def forward(self, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34):
        mul_tensor = torch.ops.aten.mul.Tensor(input_24, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_24, [2], True, dtype = None);  input_24 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_25, [2], True, dtype = None);  input_25 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_26, sum_dim_int_list_1);  input_26 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, sum_dim_int_list, alpha = 1);  mul_tensor = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_1, alpha = 1);  sub_tensor = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_27, sub_tensor_1);  input_27 = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_28, torch.bfloat16);  input_28 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_3);  convert_element_type_default = mul_tensor_3 = None
        view_default = torch.ops.aten.view.default(mul_tensor_4, input_29);  mul_tensor_4 = input_29 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_30);  input_30 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_31);  permute_default = input_31 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_32);  mm_default = input_32 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_33);  view_default_1 = input_33 = None
        slice_tensor = torch.ops.aten.slice.Tensor(view_default_2, 2, 0, 6, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(view_default_2, 2, 6, 12, 1);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(slice_tensor_1, input_34);  slice_tensor_1 = input_34 = None
        permute_default_1 = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1, 3]);  slice_tensor = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        clone_default_1 = torch.ops.aten.clone.default(view_default_3, memory_format = torch.contiguous_format);  view_default_3 = None
        return (convert_element_type_default_2, clone_default, clone_default_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
