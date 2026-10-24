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
    def forward(self, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        view_default = torch.ops.aten.view.default(input_34, input_35);  input_34 = input_35 = None
        permute_default = torch.ops.aten.permute.default(input_36, [1, 0]);  input_36 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_37);  addmm_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_1, 0.125);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        permute_default_1 = torch.ops.aten.permute.default(input_39, [1, 0]);  input_39 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default_1, input_37);  addmm_default_1 = input_37 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_40);  view_default_2 = input_40 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mul_tensor, input_41);  mul_tensor = input_41 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        clone_default = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_42);  clone_default = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_42);  clone_default_1 = input_42 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1]);  view_default_6 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_5, permute_default_4)
        view_default_7 = torch.ops.aten.view.default(bmm_default, input_43);  input_43 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_7, input_44, alpha = 1);  view_default_7 = input_44 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor, input_45);  add_tensor = input_45 = None
        amax_default = torch.ops.aten.amax.default(view_default_8, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(view_default_8, amax_default, alpha = 1);  view_default_8 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        bmm_default_1 = torch.ops.aten.bmm.default(convert_element_type_default_2, input_46);  input_46 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 2, 1]);  convert_element_type_default_2 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1]);  view_default_5 = None
        permute_default_7 = torch.ops.aten.permute.default(permute_default_4, [0, 2, 1]);  permute_default_4 = None
        return (bmm_default, amax_default, sum_dim_int_list, bmm_default_1, permute_default_5, permute_default_6, permute_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
