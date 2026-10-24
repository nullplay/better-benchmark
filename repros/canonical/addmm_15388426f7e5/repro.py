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
    def forward(self, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27):
        rsqrt_default = torch.ops.aten.rsqrt.default(input_20);  input_20 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_21, input_22, alpha = 1);  input_21 = input_22 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_23, -1);  input_23 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, -1);  unsqueeze_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, unsqueeze_default_1);  mul_tensor = unsqueeze_default_1 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(input_24, -1);  input_24 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, -1);  unsqueeze_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_3, alpha = 1);  mul_tensor_1 = unsqueeze_default_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        relu_default = torch.ops.aten.relu.default(convert_element_type_default);  convert_element_type_default = None
        mean_dim = torch.ops.aten.mean.dim(relu_default, [2, 3], False, dtype = None);  relu_default = None
        lt_scalar = torch.ops.aten.lt.Scalar(input_25, 0.8);  input_25 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(lt_scalar, torch.bfloat16);  lt_scalar = None
        div_scalar = torch.ops.aten.div.Scalar(convert_element_type_default_1, 0.8);  convert_element_type_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mean_dim, div_scalar);  mean_dim = div_scalar = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_26, torch.bfloat16);  input_26 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_27, torch.bfloat16);  input_27 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_2, mul_tensor_2, permute_default, beta = 1, alpha = 1);  convert_element_type_default_2 = mul_tensor_2 = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
