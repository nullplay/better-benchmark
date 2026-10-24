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
    def forward(self, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_43, torch.bfloat16);  input_43 = None
        view_default = torch.ops.aten.view.default(input_44, input_45);  input_44 = input_45 = None
        permute_default = torch.ops.aten.permute.default(input_46, [1, 0]);  input_46 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_47);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_48);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_49, view_default, input_50, beta = 1, alpha = 1);  input_49 = view_default = input_50 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_47);  addmm_default_1 = input_47 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_48);  view_default_3 = input_48 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 2]);  permute_default_2 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_51, implicit = False);  permute_default_1 = input_51 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_52);  clone_default = input_52 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_3, input_53, implicit = False);  permute_default_3 = input_53 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_54);  clone_default_1 = input_54 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_5, view_default_6)
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_55);  input_55 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_7, 0.125)
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1);  convert_element_type_default_2 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default, alpha = 1);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default_1, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax_default_1, alpha = 1)
        abs_default = torch.ops.aten.abs.default(convert_element_type_default_1)
        ne_scalar = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        eq_tensor = torch.ops.aten.eq.Tensor(convert_element_type_default_1, convert_element_type_default_1);  convert_element_type_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default = torch.ops.aten.logical_not.default(mul_tensor_3);  mul_tensor_3 = None
        any_dims = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1 = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        where_self = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  mul_tensor_2 = sub_tensor_1 = None
        exp_default = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1]);  view_default_5 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1]);  view_default_6 = None
        return (bmm_default, amax_default, amax_default_1, logical_not_default_1, sum_dim_int_list, convert_element_type_default_3, permute_default_4, permute_default_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
