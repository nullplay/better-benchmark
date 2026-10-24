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
    def forward(self, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28):
        mm_default = torch.ops.aten.mm.default(input_21, input_22);  input_22 = None
        permute_default = torch.ops.aten.permute.default(input_21, [1, 0]);  input_21 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_23);  permute_default = input_23 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(convert_element_type_default, 1);  convert_element_type_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        expand_default = torch.ops.aten.expand.default(unsqueeze_default_1, input_24, implicit = False);  unsqueeze_default_1 = input_24 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(div_scalar, input_25);  input_25 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [3], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(input_26, input_27, alpha = 1);  input_26 = input_27 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_28);  sub_tensor = input_28 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [3], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_scalar, mul_tensor_2);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2], False, dtype = None);  mul_tensor_5 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(div_scalar, [0, 1, 2], False, dtype = None);  div_scalar = None
        return (convert_element_type_default_1, mul_tensor_4, sub_tensor_1, sum_dim_int_list_2, sum_dim_int_list_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
