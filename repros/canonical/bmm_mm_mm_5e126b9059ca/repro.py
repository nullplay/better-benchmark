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
    def forward(self, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35):
        view_default = torch.ops.aten.view.default(input_22, input_23);  input_22 = input_23 = None
        bmm_default = torch.ops.aten.bmm.default(view_default, input_24);  view_default = input_24 = None
        view_default_1 = torch.ops.aten.view.default(bmm_default, input_25);  bmm_default = input_25 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_26);  clone_default = input_26 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_27);  view_default_2 = input_27 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default_1, input_28);  permute_default_1 = input_28 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_29);  view_default_3 = input_29 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_30);  mm_default_1 = input_30 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_31);  input_31 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_32, input_33)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_1);  convert_element_type_default = mul_tensor_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1], True, dtype = torch.float32);  mul_tensor_2 = None
        view_default_5 = torch.ops.aten.view.default(sum_dim_int_list, input_34);  sum_dim_int_list = input_34 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, input_32);  input_32 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor, input_33);  mul_tensor = input_33 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = torch.float32);  mul_tensor_3 = None
        add_tensor = torch.ops.aten.add.Tensor(input_35, mul_tensor_4, alpha = 1);  input_35 = mul_tensor_4 = None
        return (convert_element_type_default_1, view_default_5, sum_dim_int_list_1, add_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
