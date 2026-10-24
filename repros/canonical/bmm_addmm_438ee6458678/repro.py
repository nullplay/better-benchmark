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
    def forward(self, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40):
        view_default = torch.ops.aten.view.default(input_29, input_30);  input_29 = input_30 = None
        permute_default = torch.ops.aten.permute.default(view_default, [2, 0, 3, 1, 4]);  view_default = None
        unbind_int = torch.ops.aten.unbind.int(permute_default, 0);  permute_default = None
        getitem = unbind_int[2];  unbind_int = None
        view_default_1 = torch.ops.aten.view.default(input_31, input_32);  input_31 = input_32 = None
        amax_default = torch.ops.aten.amax.default(view_default_1, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(view_default_1, amax_default, alpha = 1);  view_default_1 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        expand_default = torch.ops.aten.expand.default(convert_element_type_default, input_33, implicit = False);  convert_element_type_default = input_33 = None
        view_default_2 = torch.ops.aten.view.default(expand_default, input_34);  expand_default = input_34 = None
        expand_default_1 = torch.ops.aten.expand.default(getitem, input_35, implicit = False);  getitem = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_36);  clone_default = input_36 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default_2, [0, 7, 0, 7, 0, 0], 0);  view_default_2 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(view_default_3, [0, 0, 0, 7, 0, 0], 0);  view_default_3 = None
        bmm_default = torch.ops.aten.bmm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        slice_tensor = torch.ops.aten.slice.Tensor(bmm_default, 1, 0, -7, 1);  bmm_default = None
        view_default_4 = torch.ops.aten.view.default(slice_tensor, input_35);  slice_tensor = input_35 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_37);  clone_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_39);  view_default_5 = input_39 = None
        permute_default_2 = torch.ops.aten.permute.default(input_40, [1, 0]);  input_40 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_6, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_1 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default, input_37);  addmm_default = input_37 = None
        return (view_default_6, view_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
