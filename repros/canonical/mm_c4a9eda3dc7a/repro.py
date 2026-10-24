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
    def forward(self, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39):
        add_tensor = torch.ops.aten.add.Tensor(input_28, input_29, alpha = 1);  input_28 = input_29 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_30, torch.bfloat16);  input_30 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default, alpha = 1);  add_tensor = convert_element_type_default = None
        getitem = input_31[0];  input_31 = None
        permute_default = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default = torch.ops.aten.view.default(permute_default, input_32);  permute_default = input_32 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_33);  view_default = input_33 = None
        permute_default_1 = torch.ops.aten.permute.default(input_34, [1, 0]);  input_34 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default_1);  view_default_1 = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_35);  mm_default = input_35 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_default);  convert_element_type_default_1 = rsqrt_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_36, torch.float32);  input_36 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(convert_element_type_default_2, 1.0, alpha = 1);  convert_element_type_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor_3);  mul_tensor = add_tensor_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_3, alpha = 1);  add_tensor_1 = convert_element_type_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_37);  convert_element_type_default_4 = input_37 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_5, alpha = 1);  add_tensor_4 = convert_element_type_default_5 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, input_39, alpha = 1);  add_tensor_5 = input_39 = None
        return (pow_tensor_scalar_1, mul_tensor_2, convert_element_type_default_6, add_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
