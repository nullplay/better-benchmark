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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        addmm_default = torch.ops.aten.addmm.default(input_35, input_36, input_37, beta = 1, alpha = 1);  input_35 = input_36 = input_37 = None
        view_default = torch.ops.aten.view.default(addmm_default, input_38);  addmm_default = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_39);  view_default = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 2, 1, 3]);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_40, input_38);  input_40 = input_38 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_39);  view_default_2 = input_39 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        mul_scalar = torch.ops.aten.mul.Scalar(permute_default, 0.4204482076268573);  permute_default = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 2]);  permute_default_1 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(permute_default_2, 0.4204482076268573);  permute_default_2 = None
        expand_default = torch.ops.aten.expand.default(mul_scalar, input_41, implicit = False);  mul_scalar = input_41 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_42);  clone_default = input_42 = None
        expand_default_1 = torch.ops.aten.expand.default(mul_scalar_1, input_43, implicit = False);  mul_scalar_1 = input_43 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_44);  clone_default_1 = input_44 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_4, view_default_5);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default, input_45)
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32)
        amax_default = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default, alpha = 1);  convert_element_type_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        eq_scalar = torch.ops.aten.eq.Scalar(view_default_6, -inf);  view_default_6 = None
        logical_not_default = torch.ops.aten.logical_not.default(eq_scalar);  eq_scalar = None
        any_dim = torch.ops.aten.any.dim(logical_not_default, -1, True);  logical_not_default = None
        logical_not_default_1 = torch.ops.aten.logical_not.default(any_dim);  any_dim = None
        full_default = torch.ops.aten.full.default(input_45, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_45 = None
        where_self = torch.ops.aten.where.self(logical_not_default_1, full_default, convert_element_type_default_1);  full_default = convert_element_type_default_1 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_46, 0.1);  input_46 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, where_self);  gt_scalar = where_self = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
        return (bmm_default, amax_default, sum_dim_int_list, logical_not_default_1, mul_tensor_1, permute_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
