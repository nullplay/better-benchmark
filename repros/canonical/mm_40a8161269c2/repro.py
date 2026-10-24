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
    def forward(self, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37):
        add_tensor = torch.ops.aten.add.Tensor(input_27, input_28, alpha = 1);  input_27 = input_28 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_29, 1.1111111111111112);  input_29 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_30, 1e-06, alpha = 1);  input_30 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_1, rsqrt_default);  add_tensor_1 = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_31, mul_tensor_1);  input_31 = mul_tensor_1 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_32, 0.1);  input_32 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(gt_scalar, mul_tensor_2);  gt_scalar = mul_tensor_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_34);  convert_element_type_default_1 = input_34 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_35);  mm_default = input_35 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_36);  input_36 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_2, [1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default, alpha = 1);  convert_element_type_default_2 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, log_default, alpha = 1);  sub_tensor = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float32);  convert_element_type_default_3 = None
        gather_default = torch.ops.aten.gather.default(convert_element_type_default_4, 1, input_37, sparse_grad = False);  convert_element_type_default_4 = input_37 = None
        return (view_default_1, amax_default, log_default, gather_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
