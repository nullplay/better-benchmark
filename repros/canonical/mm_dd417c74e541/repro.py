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
    def forward(self, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31):
        add_tensor = torch.ops.aten.add.Tensor(input_23, input_24, alpha = 1);  input_23 = input_24 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_25, torch.bfloat16);  input_25 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default, alpha = 1);  add_tensor = convert_element_type_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_26, input_27);  input_26 = input_27 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_1, alpha = 1);  add_tensor_1 = convert_element_type_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default);  convert_element_type_default_2 = rsqrt_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_28, torch.float32);  input_28 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(convert_element_type_default_3, 1.0, alpha = 1);  convert_element_type_default_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, add_tensor_4);  mul_tensor_1 = add_tensor_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_4, input_29);  convert_element_type_default_4 = input_29 = None
        permute_default = torch.ops.aten.permute.default(input_30, [1, 0]);  input_30 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_31);  mm_default = input_31 = None
        div_tensor = torch.ops.aten.div.Tensor(view_default_1, 30.0);  view_default_1 = None
        tanh_default = torch.ops.aten.tanh.default(div_tensor);  div_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(tanh_default, 30.0);  tanh_default = None
        return (mul_tensor_3,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
