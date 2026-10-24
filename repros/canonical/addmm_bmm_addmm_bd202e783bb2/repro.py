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
    def forward(self, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        view_default = torch.ops.aten.view.default(input_39, input_40);  input_39 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        permute_default = torch.ops.aten.permute.default(input_42, [1, 0]);  input_42 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_43);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_44);  view_default_1 = input_44 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(input_45, input_46, alpha = 1);  input_45 = input_46 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_1, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax_default, alpha = 1);  convert_element_type_default_1 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        eq_scalar = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1 = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default = torch.ops.aten.full.default(input_47, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self = torch.ops.aten.where.self(logical_not_default_1, full_default, convert_element_type_default_2);  logical_not_default_1 = full_default = convert_element_type_default_2 = None
        expand_default = torch.ops.aten.expand.default(where_self, input_47, implicit = False);  where_self = input_47 = None
        view_default_3 = torch.ops.aten.view.default(expand_default, input_48);  expand_default = input_48 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_1, input_49, implicit = False);  permute_default_1 = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_50);  clone_default = input_50 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_3, view_default_4);  view_default_3 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default, input_49);  bmm_default = input_49 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_51);  clone_default_1 = input_51 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_52, torch.bfloat16);  input_52 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_53, torch.bfloat16);  input_53 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_40);  view_default_6 = input_40 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_7, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_1, input_43);  input_43 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_54, view_default_8);  input_54 = view_default_8 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
        return (view_default_7, addmm_default_1, mul_tensor, permute_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
