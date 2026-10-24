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
    def forward(self, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49):
        addmm_default = torch.ops.aten.addmm.default(input_37, input_38, input_39, beta = 1, alpha = 1);  input_37 = input_38 = input_39 = None
        view_default = torch.ops.aten.view.default(addmm_default, input_40);  addmm_default = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_41);  view_default = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_42, input_40);  input_42 = input_40 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_41);  view_default_2 = input_41 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        full_default = torch.ops.aten.full.default(input_43, -inf, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1 = torch.ops.aten.full.default(input_43, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_43 = None
        where_self = torch.ops.aten.where.self(input_44, full_default_1, full_default);  input_44 = full_default_1 = full_default = None
        mul_scalar = torch.ops.aten.mul.Scalar(permute_default, 0.334370152488211);  permute_default = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(permute_default_2, 0.334370152488211);  permute_default_2 = None
        expand_default = torch.ops.aten.expand.default(mul_scalar, input_45, implicit = False);  mul_scalar = input_45 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_46);  clone_default = input_46 = None
        expand_default_1 = torch.ops.aten.expand.default(mul_scalar_1, input_47, implicit = False);  mul_scalar_1 = input_47 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_48);  clone_default_1 = input_48 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_4, view_default_5);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default, input_49);  bmm_default = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_6, where_self, alpha = 1);  view_default_6 = where_self = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        amax_default = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default, alpha = 1);  convert_element_type_default = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        eq_scalar = torch.ops.aten.eq.Scalar(add_tensor, -inf);  add_tensor = None
        logical_not_default = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1 = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default_2 = torch.ops.aten.full.default(input_49, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_self_1 = torch.ops.aten.where.self(logical_not_default_1, full_default_2, convert_element_type_default_1);  logical_not_default_1 = full_default_2 = convert_element_type_default_1 = None
        expand_default_2 = torch.ops.aten.expand.default(where_self_1, input_49, implicit = False);  input_49 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
        return (where_self_1, expand_default_2, permute_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
