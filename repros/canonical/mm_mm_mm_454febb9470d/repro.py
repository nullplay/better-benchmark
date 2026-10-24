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
    def forward(self, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56):
        add_tensor = torch.ops.aten.add.Tensor(input_41, input_42, alpha = 1);  input_41 = input_42 = None
        view_default = torch.ops.aten.view.default(input_43, input_44);  input_43 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        mm_default = torch.ops.aten.mm.default(input_45, input_46);  input_45 = input_46 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_44);  mm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_47, convert_element_type_default_1);  input_47 = convert_element_type_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_1, input_48)
        permute_default = torch.ops.aten.permute.default(input_49, [1, 0]);  input_49 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, permute_default);  view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_50);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        neg_default = torch.ops.aten.neg.default(convert_element_type_default_2)
        exp_default = torch.ops.aten.exp.default(neg_default);  neg_default = None
        add_tensor_4 = torch.ops.aten.add.Tensor(exp_default, 1, alpha = 1);  exp_default = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_2, add_tensor_4);  convert_element_type_default_2 = add_tensor_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        view_default_4 = torch.ops.aten.view.default(mul_tensor_1, input_48);  mul_tensor_1 = input_48 = None
        permute_default_1 = torch.ops.aten.permute.default(input_51, [1, 0]);  input_51 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, permute_default_1);  view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_50);  mm_default_2 = input_50 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, view_default_5);  convert_element_type_default_3 = view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(mul_tensor_2, input_52);  mul_tensor_2 = input_52 = None
        permute_default_2 = torch.ops.aten.permute.default(input_53, [1, 0]);  input_53 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_6, permute_default_2);  view_default_6 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_3, input_44);  mm_default_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_7, alpha = 1);  add_tensor_2 = view_default_7 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_54);  convert_element_type_default_4 = input_54 = None
        view_default_8 = torch.ops.aten.view.default(input_55, input_44);  input_55 = input_44 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_8, alpha = 1);  add_tensor_5 = view_default_8 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, input_56, alpha = 1);  add_tensor_6 = input_56 = None
        return (pow_tensor_scalar_1, mul_tensor_3, convert_element_type_default_5, add_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
