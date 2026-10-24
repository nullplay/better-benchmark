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
    def forward(self, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49):
        add_tensor = torch.ops.aten.add.Tensor(input_33, input_34, alpha = 1);  input_33 = input_34 = None
        view_default = torch.ops.aten.view.default(input_35, input_36);  input_35 = input_36 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0, 2, 4, 3]);  view_default = None
        unbind_int = torch.ops.aten.unbind.int(permute_default, 0);  permute_default = None
        getitem = unbind_int[2];  unbind_int = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_37, 0.125);  input_37 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_38, input_39, alpha = 1);  input_38 = input_39 = None
        logical_not_default = torch.ops.aten.logical_not.default(input_40);  input_40 = None
        where_self = torch.ops.aten.where.self(logical_not_default, mul_tensor, sub_tensor);  logical_not_default = mul_tensor = sub_tensor = None
        exp_default = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        expand_default = torch.ops.aten.expand.default(convert_element_type_default, input_41, implicit = False);  convert_element_type_default = input_41 = None
        view_default_1 = torch.ops.aten.view.default(expand_default, input_42);  expand_default = input_42 = None
        expand_default_1 = torch.ops.aten.expand.default(getitem, input_43, implicit = False);  getitem = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_44);  clone_default = input_44 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default_1, [0, 4, 0, 4, 0, 0], 0);  view_default_1 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(view_default_2, [0, 0, 0, 4, 0, 0], 0);  view_default_2 = None
        bmm_default = torch.ops.aten.bmm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        slice_tensor = torch.ops.aten.slice.Tensor(bmm_default, 1, 0, -4, 1);  bmm_default = None
        view_default_3 = torch.ops.aten.view.default(slice_tensor, input_43);  slice_tensor = input_43 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 1, 3, 2]);  view_default_3 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(clone_default_1, input_45);  clone_default_1 = input_45 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_46, torch.bfloat16);  input_46 = None
        convolution_default = torch.ops.aten.convolution.default(view_default_4, convert_element_type_default_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convolution_default, alpha = 1);  add_tensor = convolution_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [0, 2, 3], correction = 0, keepdim = True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_1, input_47, alpha = 1);  input_47 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, input_48, alpha = 1);  input_48 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_1, input_49, alpha = 1);  add_tensor_1 = input_49 = None
        return (view_default_4, var_mean_correction, sub_tensor_1, add_tensor_2, sub_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
