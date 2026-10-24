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
    def forward(self, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42):
        add_tensor = torch.ops.aten.add.Tensor(input_28, input_29, alpha = 1);  input_28 = input_29 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_30);  input_30 = None
        permute_default = torch.ops.aten.permute.default(input_31, [1, 0]);  input_31 = None
        addmm_default = torch.ops.aten.addmm.default(input_32, view_default, permute_default, beta = 1, alpha = 1);  input_32 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_33);  addmm_default = input_33 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_34);  input_34 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default, [0, 2, 1]);  convert_element_type_default = None
        convolution_default = torch.ops.aten.convolution.default(permute_default_2, input_35, None, [1], [4], [1], False, [0], 768);  permute_default_2 = input_35 = None
        convolution_default_1 = torch.ops.aten.convolution.default(convolution_default, input_36, None, [1], [0], [1], False, [0], 1);  convolution_default = input_36 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convolution_default_1, input_37, alpha = 1);  convolution_default_1 = input_37 = None
        permute_default_3 = torch.ops.aten.permute.default(add_tensor_1, [0, 2, 1]);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(permute_default_3, view_default_1);  permute_default_3 = view_default_1 = None
        clone_default = torch.ops.aten.clone.default(mul_tensor, memory_format = torch.contiguous_format);  mul_tensor = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_38);  clone_default = input_38 = None
        permute_default_4 = torch.ops.aten.permute.default(input_39, [1, 0]);  input_39 = None
        mm_default = torch.ops.aten.mm.default(view_default_3, permute_default_4);  view_default_3 = permute_default_4 = None
        view_default_4 = torch.ops.aten.view.default(mm_default, input_40);  mm_default = input_40 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_4, input_41, alpha = 1);  view_default_4 = input_41 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_2, input_42);  add_tensor_2 = input_42 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_1, [1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax_default, alpha = 1);  convert_element_type_default_1 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        return (permute_default_1, convert_element_type_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
