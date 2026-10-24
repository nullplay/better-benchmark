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
    def forward(self, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        mul_tensor = torch.ops.aten.mul.Tensor(input_40, input_41);  input_40 = input_41 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_42, input_43);  input_42 = input_43 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_3, alpha = 1);  mul_tensor_1 = mul_tensor_3 = None
        view_default = torch.ops.aten.view.default(input_44, input_45);  input_44 = input_45 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_46, 0.1);  input_46 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1.1111111111111112);  mul_tensor_4 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_5, alpha = 1);  add_tensor = mul_tensor_5 = None
        gt_scalar_1 = torch.ops.aten.gt.Scalar(input_47, 0.1);  input_47 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(gt_scalar_1, add_tensor_1);  gt_scalar_1 = add_tensor_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, 1.1111111111111112);  mul_tensor_6 = None
        select_int = torch.ops.aten.select.int(mul_tensor_7, 1, 0)
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_48, torch.bfloat16);  input_48 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_49, torch.bfloat16);  input_49 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(select_int, torch.bfloat16);  select_int = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, convert_element_type_default_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default = convert_element_type_default_2 = permute_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(addmm_default, torch.float32);  addmm_default = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_3, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_3, amax_default, alpha = 1);  convert_element_type_default_3 = amax_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_6, input_52);  convert_element_type_default_6 = input_52 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default_1, [0, 3, 0, 0], 0);  permute_default_1 = None
        full_default = torch.ops.aten.full.default(input_53, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_53 = None
        cat_default = torch.ops.aten.cat.default([convert_element_type_default_4, full_default], 0);  convert_element_type_default_4 = full_default = None
        addmm_default_1 = torch.ops.aten.addmm.default(cat_default, view_default_1, constant_pad_nd_default, beta = 1, alpha = 1);  cat_default = view_default_1 = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(addmm_default_1, 1, 0, -3, 1);  addmm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(slice_tensor, input_54);  slice_tensor = input_54 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default_7, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_7, amax_default_1, alpha = 1);  convert_element_type_default_7 = amax_default_1 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, log_default, alpha = 1);  sub_tensor_1 = log_default = None
        return (sub_tensor, sub_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
