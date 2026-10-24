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
    def forward(self, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        mul_tensor = torch.ops.aten.mul.Tensor(input_26, input_27);  input_26 = input_27 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_28);  mul_tensor = input_28 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_29);  mul_tensor_1 = input_29 = None
        expand_default = torch.ops.aten.expand.default(input_30, input_31, implicit = False);  input_30 = input_31 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_32, 1.0);  input_32 = None
        mul_scalar = torch.ops.aten.mul.Scalar(pow_tensor_scalar, 2.0);  pow_tensor_scalar = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar);  div_scalar = mul_scalar = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3, alpha = 1);  mul_tensor_2 = mul_tensor_3 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_4);  convert_element_type_default = mul_tensor_4 = None
        view_default = torch.ops.aten.view.default(mul_tensor_5, input_34);  mul_tensor_5 = input_34 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_35);  permute_default = input_35 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_36);  view_default = input_36 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_37);  mm_default_1 = input_37 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, 1.1111111111111112);  convert_element_type_default_4 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_6);  convert_element_type_default_3 = mul_tensor_6 = None
        full_default = torch.ops.aten.full.default(input_39, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_39 = None
        where_self = torch.ops.aten.where.self(input_40, full_default, mul_tensor_7);  input_40 = full_default = mul_tensor_7 = None
        view_default_2 = torch.ops.aten.view.default(where_self, input_41);  where_self = input_41 = None
        return (convert_element_type_default_2, view_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
