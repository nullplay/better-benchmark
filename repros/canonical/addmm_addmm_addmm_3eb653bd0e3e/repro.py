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
    def forward(self, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_37, torch.bfloat16);  input_37 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_38);  convert_element_type_default = None
        addmm_default = torch.ops.aten.addmm.default(input_39, view_default, input_40, beta = 1, alpha = 1);  input_39 = input_40 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_41, view_default, input_42, beta = 1, alpha = 1);  input_41 = input_42 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_43, torch.bfloat16);  input_43 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_44, torch.bfloat16);  input_44 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default_2, input_45);  addmm_default_2 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_46);  view_default_1 = input_46 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_47);  clone_default = input_47 = None
        maximum_default = torch.ops.aten.maximum.default(input_48, input_49);  input_48 = input_49 = None
        view_default_4 = torch.ops.aten.view.default(maximum_default, input_50);  maximum_default = input_50 = None
        amax_default = torch.ops.aten.amax.default(view_default_4, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(view_default_4, amax_default, alpha = 1);  view_default_4 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_51, 0.1);  input_51 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, div_tensor);  gt_scalar = div_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        bmm_default = torch.ops.aten.bmm.default(convert_element_type_default_3, view_default_3);  convert_element_type_default_3 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default, input_52);  bmm_default = input_52 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_45);  clone_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_53, torch.bfloat16);  input_53 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_54, torch.bfloat16);  input_54 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_38);  view_default_6 = input_38 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_7, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_45);  input_45 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_55, view_default_8);  input_55 = view_default_8 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1]);  view_default_3 = None
        return (addmm_default, addmm_default_1, view_default_7, addmm_default_3, mul_tensor_2, permute_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
