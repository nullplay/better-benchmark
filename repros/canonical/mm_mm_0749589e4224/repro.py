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
    def forward(self, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37):
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_26, 1);  input_26 = None
        expand_default = torch.ops.aten.expand.default(unsqueeze_default, input_27, implicit = False);  unsqueeze_default = input_27 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 196);  expand_default = None
        full_default = torch.ops.aten.full.default(input_28, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        slice_scatter_default = torch.ops.aten.slice_scatter.default(full_default, div_scalar, 1, 1, 9223372036854775807, 1);  full_default = div_scalar = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_29, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_29, [2], True, dtype = None);  input_29 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_30, [2], True, dtype = None);  input_30 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_31, sum_dim_int_list_1);  input_31 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, sum_dim_int_list, alpha = 1);  mul_tensor = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_1, alpha = 1);  sub_tensor = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_32, sub_tensor_1);  input_32 = sub_tensor_1 = None
        add_tensor = torch.ops.aten.add.Tensor(slice_scatter_default, mul_tensor_2, alpha = 1);  slice_scatter_default = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor, input_33);  add_tensor = input_33 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_34);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_35);  input_35 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(input_36, [0, 2, 1, 3]);  input_36 = None
        view_default_1 = torch.ops.aten.view.default(permute_default_1, input_28);  permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_34);  view_default_1 = input_34 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, view_default_2);  permute_default = view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_28);  mm_default = input_28 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_37);  view_default_3 = input_37 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        return (convert_element_type_default_1, permute_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
