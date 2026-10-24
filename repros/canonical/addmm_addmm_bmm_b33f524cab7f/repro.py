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
    def forward(self, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_31, 1);  input_31 = None
        eq_scalar = torch.ops.aten.eq.Scalar(unsqueeze_default, 0);  unsqueeze_default = None
        full_default = torch.ops.aten.full.default(input_32, -998244352.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_32 = None
        view_default = torch.ops.aten.view.default(input_33, input_34);  input_33 = input_34 = None
        permute_default = torch.ops.aten.permute.default(input_35, [1, 0]);  input_35 = None
        addmm_default = torch.ops.aten.addmm.default(input_36, view_default, permute_default, beta = 1, alpha = 1);  input_36 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_37);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_38);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_39, implicit = False);  permute_default_1 = input_39 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_40);  clone_default = input_40 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_41, input_42, input_43, beta = 1, alpha = 1);  input_41 = input_42 = input_43 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_37);  addmm_default_1 = input_37 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_38);  view_default_4 = input_38 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 3, 2]);  permute_default_2 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_3, input_44, implicit = False);  permute_default_3 = input_44 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_45);  clone_default_1 = input_45 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_3, view_default_6);  view_default_3 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_46);  bmm_default = input_46 = None
        div_tensor = torch.ops.aten.div.Tensor(view_default_7, 8.0);  view_default_7 = None
        where_self = torch.ops.aten.where.self(eq_scalar, full_default, div_tensor);  eq_scalar = full_default = div_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default, alpha = 1);  convert_element_type_default = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor_1 = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None
        return (convert_element_type_default_1,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
