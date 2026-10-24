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
        mm_default = torch.ops.aten.mm.default(input_27, input_28);  input_28 = None
        permute_default = torch.ops.aten.permute.default(input_27, [1, 0]);  input_27 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_29);  permute_default = input_29 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_30, torch.bfloat16);  input_30 = None
        div_scalar = torch.ops.aten.div.Scalar(convert_element_type_default_1, 0.8);  convert_element_type_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(mm_default, div_scalar);  mm_default = div_scalar = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_31, torch.float32);  input_31 = None
        le_scalar = torch.ops.aten.le.Scalar(convert_element_type_default_3, -3)
        lt_scalar = torch.ops.aten.lt.Scalar(convert_element_type_default_3, 3)
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_3, 3);  convert_element_type_default_3 = None
        add_tensor = torch.ops.aten.add.Tensor(div_tensor, 0.5, alpha = 1);  div_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, add_tensor);  add_tensor = None
        where_self = torch.ops.aten.where.self(lt_scalar, mul_tensor_1, convert_element_type_default_2);  lt_scalar = mul_tensor_1 = convert_element_type_default_2 = None
        full_default = torch.ops.aten.full.default(input_32, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_32 = None
        where_self_1 = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(where_self_1, torch.bfloat16);  where_self_1 = None
        mm_default_2 = torch.ops.aten.mm.default(convert_element_type_default_4, input_33);  input_33 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_34);  permute_default_1 = input_34 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0], True, dtype = torch.float32);  convert_element_type_default_4 = None
        view_default = torch.ops.aten.view.default(sum_dim_int_list, input_35);  sum_dim_int_list = input_35 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_2, input_36);  mm_default_2 = input_36 = None
        expand_default = torch.ops.aten.expand.default(view_default_1, input_37, implicit = False);  view_default_1 = input_37 = None
        return (convert_element_type_default, convert_element_type_default_5, convert_element_type_default_6, expand_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
