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
    def forward(self, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36):
        mm_default = torch.ops.aten.mm.default(input_25, input_26);  input_26 = None
        permute_default = torch.ops.aten.permute.default(input_25, [1, 0]);  input_25 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_27);  permute_default = input_27 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_28, torch.bfloat16);  input_28 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.25);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mm_default, mul_tensor);  mm_default = mul_tensor = None
        view_default = torch.ops.aten.view.default(mul_tensor_1, input_29);  mul_tensor_1 = input_29 = None
        expand_default = torch.ops.aten.expand.default(view_default, input_30, implicit = False);  view_default = input_30 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_31, torch.bfloat16);  input_31 = None
        le_scalar = torch.ops.aten.le.Scalar(convert_element_type_default_2, 0.0)
        ge_scalar = torch.ops.aten.ge.Scalar(convert_element_type_default_2, 6.0);  convert_element_type_default_2 = None
        bitwise_or_tensor = torch.ops.aten.bitwise_or.Tensor(le_scalar, ge_scalar);  le_scalar = ge_scalar = None
        full_default = torch.ops.aten.full.default(input_32, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_32 = None
        where_self = torch.ops.aten.where.self(bitwise_or_tensor, full_default, div_scalar);  bitwise_or_tensor = full_default = div_scalar = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_3, [0, 2, 3], False, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(input_33, input_34, alpha = 1);  input_33 = input_34 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, sub_tensor)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3], False, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00021258503401360543)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor, input_35);  sub_tensor = input_35 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, mul_tensor_4, alpha = 1);  convert_element_type_default_3 = mul_tensor_4 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, input_36, alpha = 1);  sub_tensor_1 = input_36 = None
        return (convert_element_type_default, sum_dim_int_list, sum_dim_int_list_1, mul_tensor_3, sub_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
