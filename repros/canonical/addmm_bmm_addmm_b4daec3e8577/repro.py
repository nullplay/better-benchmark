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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_30, torch.bfloat16);  input_30 = None
        add_tensor = torch.ops.aten.add.Tensor(input_31, input_32, alpha = 1);  input_31 = input_32 = None
        full_default = torch.ops.aten.full.default(input_33, -3.3895313892515355e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_33 = None
        maximum_default = torch.ops.aten.maximum.default(add_tensor, full_default);  add_tensor = full_default = None
        view_default = torch.ops.aten.view.default(maximum_default, input_34);  maximum_default = input_34 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_1, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax_default, alpha = 1);  convert_element_type_default_1 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default, input_35);  convert_element_type_default = None
        permute_default = torch.ops.aten.permute.default(input_36, [1, 0]);  input_36 = None
        addmm_default = torch.ops.aten.addmm.default(input_37, view_default_1, permute_default, beta = 1, alpha = 1);  input_37 = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_38);  addmm_default = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_39);  view_default_2 = input_39 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_40);  clone_default = input_40 = None
        bmm_default = torch.ops.aten.bmm.default(convert_element_type_default_2, view_default_4);  convert_element_type_default_2 = view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default, input_41);  bmm_default = input_41 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_38);  clone_default_1 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_35);  view_default_6 = input_35 = None
        permute_default_3 = torch.ops.aten.permute.default(input_42, [1, 0]);  input_42 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_43, view_default_7, permute_default_3, beta = 1, alpha = 1);  input_43 = view_default_7 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_1, input_38);  addmm_default_1 = input_38 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_44, view_default_8, alpha = 1);  input_44 = view_default_8 = None
        return (add_tensor_1,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
