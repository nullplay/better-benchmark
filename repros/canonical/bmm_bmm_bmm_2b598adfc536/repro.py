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
    def forward(self, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57):
        add_tensor = torch.ops.aten.add.Tensor(input_37, input_38, alpha = 1);  input_37 = input_38 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        convolution_backward_default = torch.ops.aten.convolution_backward.default(convert_element_type_default, input_39, input_40, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_default = input_39 = input_40 = None
        getitem = convolution_backward_default[0];  convolution_backward_default = None
        view_default = torch.ops.aten.view.default(getitem, input_41);  getitem = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_42);  clone_default = input_42 = None
        bmm_default = torch.ops.aten.bmm.default(input_43, view_default_1);  input_43 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_1, input_44);  view_default_1 = input_44 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default, input_45);  bmm_default = None
        view_default_3 = torch.ops.aten.view.default(bmm_default_1, input_46);  bmm_default_1 = input_46 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_47, torch.float32);  input_47 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_48, input_49, alpha = 1);  input_48 = input_49 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, 0.08838834764831845);  sub_tensor = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, input_50, alpha = 1);  convert_element_type_default_2 = input_50 = None
        where_self = torch.ops.aten.where.self(input_51, mul_tensor, sub_tensor_1);  input_51 = mul_tensor = sub_tensor_1 = None
        exp_default = torch.ops.aten.exp.default(where_self);  where_self = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_52);  exp_default = input_52 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, div_tensor);  convert_element_type_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_1);  neg_default = sum_dim_int_list = mul_tensor_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 0.08838834764831845);  convert_element_type_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mul_tensor_2, input_53);  mul_tensor_2 = input_53 = None
        bmm_default_2 = torch.ops.aten.bmm.default(input_54, view_default_4);  input_54 = None
        bmm_default_3 = torch.ops.aten.bmm.default(view_default_4, input_55);  view_default_4 = input_55 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default_2, input_41);  bmm_default_2 = input_41 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default_3, input_45);  bmm_default_3 = input_45 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_5, [0, 1, 3, 2]);  view_default_5 = None
        cat_default = torch.ops.aten.cat.default([view_default_6, permute_default_1, view_default_2], 0);  view_default_6 = permute_default_1 = view_default_2 = None
        view_default_7 = torch.ops.aten.view.default(cat_default, input_56);  cat_default = input_56 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_7, [1, 0, 2, 4, 3]);  view_default_7 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_1, input_57);  clone_default_1 = input_57 = None
        return (view_default_8,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
