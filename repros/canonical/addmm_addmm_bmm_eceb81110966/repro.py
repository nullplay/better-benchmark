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
    def forward(self, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44):
        full_default = torch.ops.aten.full.default(input_31, False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_31 = None
        full_default_1 = torch.ops.aten.full.default(input_32, -3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_34);  convert_element_type_default = input_34 = None
        permute_default = torch.ops.aten.permute.default(input_35, [1, 0]);  input_35 = None
        addmm_default = torch.ops.aten.addmm.default(input_36, view_default, permute_default, beta = 1, alpha = 1);  input_36 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_37);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_38);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_39);  clone_default = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_40, input_41, input_42, beta = 1, alpha = 1);  input_40 = input_41 = input_42 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_37);  addmm_default_1 = input_37 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_38);  view_default_4 = input_38 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_39);  clone_default_1 = input_39 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1]);  view_default_6 = None
        full_default_2 = torch.ops.aten.full.default(input_32, 8.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cpu'), pin_memory = False);  input_32 = None
        div_tensor = torch.ops.aten.div.Tensor(permute_default_3, full_default_2);  permute_default_3 = full_default_2 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_3, div_tensor);  view_default_3 = div_tensor = None
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_43);  bmm_default = input_43 = None
        where_self = torch.ops.aten.where.self(full_default, full_default_1, view_default_7);  full_default = full_default_1 = view_default_7 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_1, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax_default, alpha = 1);  convert_element_type_default_1 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor_1 = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(div_tensor_1, torch.bfloat16);  div_tensor_1 = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_2, input_44);  convert_element_type_default_2 = input_44 = None
        return (view_default_8,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
