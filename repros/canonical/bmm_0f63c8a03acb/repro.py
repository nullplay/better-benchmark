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
    def forward(self, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_31, torch.float32);  input_31 = None
        any_dims = torch.ops.aten.any.dims(input_32, [3], True);  input_32 = None
        logical_not_default = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_33, input_34, alpha = 1);  input_33 = input_34 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default, [3], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default, alpha = 1);  convert_element_type_default = amax_default = None
        where_self = torch.ops.aten.where.self(logical_not_default, mul_tensor, sub_tensor_1);  logical_not_default = mul_tensor = sub_tensor_1 = None
        exp_default = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [3], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 4);  convert_element_type_default_1 = None
        view_default = torch.ops.aten.view.default(unsqueeze_default, input_35);  unsqueeze_default = input_35 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_36, 0);  input_36 = None
        view_default_1 = torch.ops.aten.view.default(unsqueeze_default_1, input_37);  unsqueeze_default_1 = input_37 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 4, 2]);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(permute_default, input_38);  permute_default = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(view_default_2, 4);  view_default_2 = None
        permute_default_1 = torch.ops.aten.permute.default(unsqueeze_default_2, [4, 1, 2, 3, 0]);  unsqueeze_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default_1, [1, 2, 4, 3, 0]);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(permute_default_2, input_39);  permute_default_2 = input_39 = None
        bmm_default = torch.ops.aten.bmm.default(view_default, view_default_3);  view_default = view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_40);  bmm_default = input_40 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [2, 0, 1, 4, 3]);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(permute_default_3, input_38);  permute_default_3 = input_38 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(view_default_5, 4);  view_default_5 = None
        permute_default_4 = torch.ops.aten.permute.default(unsqueeze_default_3, [0, 1, 4, 3, 2]);  unsqueeze_default_3 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 1, 3, 4, 2]);  permute_default_4 = None
        clone_default = torch.ops.aten.clone.default(permute_default_5, memory_format = torch.contiguous_format);  permute_default_5 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_41);  clone_default = input_41 = None
        return (view_default_6,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
