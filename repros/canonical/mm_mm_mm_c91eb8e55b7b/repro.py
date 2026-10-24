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
    def forward(self, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70):
        mul_tensor = torch.ops.aten.mul.Tensor(input_47, input_48);  input_47 = input_48 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_49);  mul_tensor = input_49 = None
        div_scalar = torch.ops.aten.div.Scalar(input_50, 512);  input_50 = None
        mul_scalar = torch.ops.aten.mul.Scalar(input_51, 2.0);  input_51 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar);  div_scalar = mul_scalar = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2, alpha = 1);  mul_tensor_1 = mul_tensor_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default, input_52);  convert_element_type_default = input_52 = None
        view_default = torch.ops.aten.view.default(mul_tensor_3, input_53);  mul_tensor_3 = input_53 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_54);  view_default = input_54 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_55);  mm_default = input_55 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_56, torch.bfloat16);  input_56 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 1.1111111111111112);  convert_element_type_default_2 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mul_tensor_4);  convert_element_type_default_1 = mul_tensor_4 = None
        full_default = torch.ops.aten.full.default(input_57, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_57 = None
        where_self = torch.ops.aten.where.self(input_58, full_default, mul_tensor_5);  input_58 = full_default = mul_tensor_5 = None
        view_default_2 = torch.ops.aten.view.default(where_self, input_59);  where_self = input_59 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_60);  permute_default = input_60 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_2, input_61);  view_default_2 = input_61 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_2, input_62);  mm_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_63);  input_63 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_64, input_65)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_7);  convert_element_type_default_3 = mul_tensor_7 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1], True, dtype = torch.float32);  mul_tensor_8 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_66);  sum_dim_int_list = input_66 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_64)
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_6, input_65);  mul_tensor_6 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [2], True, dtype = torch.float32);  mul_tensor_9 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_10, alpha = 1);  add_tensor = mul_tensor_10 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_65, 3);  input_65 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_scalar_1, pow_tensor_scalar);  mul_scalar_1 = pow_tensor_scalar = None
        expand_default = torch.ops.aten.expand.default(mul_tensor_11, input_62, implicit = False);  mul_tensor_11 = input_62 = None
        div_scalar_1 = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(input_64, 1.0);  input_64 = None
        mul_scalar_2 = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_2);  div_scalar_1 = mul_scalar_2 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_12, alpha = 1);  add_tensor_1 = mul_tensor_12 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16)
        mul_tensor_13 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, input_67);  convert_element_type_default_5 = input_67 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_68, input_69);  input_68 = input_69 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_14, alpha = 1);  add_tensor_2 = mul_tensor_14 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, input_70, alpha = 1);  add_tensor_3 = input_70 = None
        return (convert_element_type_default_4, view_default_4, mul_tensor_13, add_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
