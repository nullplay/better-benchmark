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
    def forward(self, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44):
        mul_tensor = torch.ops.aten.mul.Tensor(input_29, 0.125);  input_29 = None
        full_default = torch.ops.aten.full.default(input_30, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_30 = None
        index_add_default = torch.ops.aten.index_add.default(full_default, 3, input_31, mul_tensor, alpha = 1);  full_default = input_31 = mul_tensor = None
        view_default = torch.ops.aten.view.default(index_add_default, input_32);  index_add_default = input_32 = None
        full_default_1 = torch.ops.aten.full.default(input_33, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_33 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(full_default_1, view_default, 2, 1, 9223372036854775807, 1);  full_default_1 = view_default = None
        view_default_1 = torch.ops.aten.view.default(slice_scatter_default, input_34);  slice_scatter_default = input_34 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_35);  view_default_1 = input_35 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 1, 2, 4, 3]);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(permute_default, input_36);  permute_default = input_36 = None
        bmm_default = torch.ops.aten.bmm.default(input_37, view_default_3);  input_37 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_38);  view_default_3 = input_38 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_39);  bmm_default = input_39 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 1, 4, 3, 2]);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default_1, input_40);  bmm_default_1 = input_40 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 1, 2, 4, 3]);  view_default_5 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_1, [3, 0, 1, 4, 2]);  permute_default_1 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(permute_default_3, 4);  permute_default_3 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_2, [2, 0, 1, 4, 3]);  permute_default_2 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(permute_default_4, 4);  permute_default_4 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(squeeze_dim_1, torch.float32);  squeeze_dim_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.bfloat16)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default, [0, 1], True, dtype = torch.float32);  convert_element_type_default = None
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list, input_41);  sum_dim_int_list = input_41 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default_1, convert_element_type_default_2, alpha = 1);  convert_element_type_default_1 = convert_element_type_default_2 = None
        view_default_7 = torch.ops.aten.view.default(squeeze_dim, input_43);  squeeze_dim = input_43 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_7, [0, 1, 4, 2, 3]);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor, input_44);  add_tensor = input_44 = None
        return (view_default_6, permute_default_5, view_default_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
