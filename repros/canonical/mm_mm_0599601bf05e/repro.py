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
    def forward(self, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27):
        mm_default = torch.ops.aten.mm.default(input_19, input_20);  input_20 = None
        permute_default = torch.ops.aten.permute.default(input_19, [1, 0]);  input_19 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_21);  permute_default = input_21 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_22, torch.bfloat16);  input_22 = None
        div_scalar = torch.ops.aten.div.Scalar(convert_element_type_default_1, 0.8);  convert_element_type_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(mm_default, div_scalar);  mm_default = div_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(mul_tensor, 2);  mul_tensor = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 3);  unsqueeze_default = None
        expand_default = torch.ops.aten.expand.default(unsqueeze_default_1, input_23, implicit = False);  unsqueeze_default_1 = input_23 = None
        div_scalar_1 = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        relu_default = torch.ops.aten.relu.default(input_24);  input_24 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default = torch.ops.aten.full.default(input_25, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_25 = None
        where_self = torch.ops.aten.where.self(le_scalar, full_default, div_scalar_1);  le_scalar = full_default = div_scalar_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 2, 3], False, dtype = None)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_26);  input_26 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_2, input_27, alpha = 1);  convert_element_type_default_2 = input_27 = None
        return (convert_element_type_default, sum_dim_int_list, mul_tensor_1, sub_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
