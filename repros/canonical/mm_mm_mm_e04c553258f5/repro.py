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
    def forward(self, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        add_tensor = torch.ops.aten.add.Tensor(input_39, input_40, alpha = 1);  input_39 = input_40 = None
        view_default = torch.ops.aten.view.default(input_41, input_42);  input_41 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        mm_default = torch.ops.aten.mm.default(input_43, input_44);  input_43 = input_44 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_42);  mm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        rsqrt_default = torch.ops.aten.rsqrt.default(input_45);  input_45 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_2, rsqrt_default);  rsqrt_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        view_default_2 = torch.ops.aten.view.default(input_46, input_47);  input_46 = None
        permute_default = torch.ops.aten.permute.default(input_48, [1, 0]);  input_48 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, permute_default);  view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_42);  mm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2);  convert_element_type_default_2 = None
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_3, rsqrt_default_1);  rsqrt_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_49, convert_element_type_default_3);  input_49 = convert_element_type_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mul_tensor_2, input_47);  mul_tensor_2 = input_47 = None
        permute_default_1 = torch.ops.aten.permute.default(input_50, [1, 0]);  input_50 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, permute_default_1);  view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_51);  mm_default_2 = input_51 = None
        relu_default = torch.ops.aten.relu.default(view_default_5);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(relu_default, input_52);  relu_default = input_52 = None
        permute_default_2 = torch.ops.aten.permute.default(input_53, [1, 0]);  input_53 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_6, permute_default_2);  view_default_6 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_3, input_42);  mm_default_3 = input_42 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_7, alpha = 1);  add_tensor_3 = view_default_7 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2);  convert_element_type_default_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(input_54);  input_54 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_5, rsqrt_default_2);  add_tensor_5 = rsqrt_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        return (pow_tensor_scalar, convert_element_type_default_1, pow_tensor_scalar_2, convert_element_type_default_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
