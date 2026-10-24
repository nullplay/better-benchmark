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
    def forward(self, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66):
        mm_default = torch.ops.aten.mm.default(input_49, input_50);  input_49 = input_50 = None
        view_default = torch.ops.aten.view.default(mm_default, input_51);  mm_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_52, torch.float32);  input_52 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        view_default_1 = torch.ops.aten.view.default(input_53, input_51);  input_53 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_1, 0.5)
        add_tensor = torch.ops.aten.add.Tensor(view_default_1, input_54, alpha = 1);  view_default_1 = input_54 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor, 0.7978845608028654);  add_tensor = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_1)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_4);  mul_tensor_4 = None
        view_default_2 = torch.ops.aten.view.default(input_55, input_51);  input_55 = input_51 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_1, view_default_2);  mul_tensor_1 = view_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_2, input_56);  convert_element_type_default_2 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_57);  permute_default = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_3, input_58);  view_default_3 = input_58 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_59);  mm_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, mul_tensor_2);  mul_tensor_2 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_6, add_tensor_1);  mul_tensor_6 = add_tensor_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_60);  mul_tensor_7 = input_60 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_9, 0.7978845608028654);  mul_tensor_9 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16)
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_61, input_62);  input_61 = input_62 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_11, torch.bfloat16);  mul_tensor_11 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default_6, convert_element_type_default_7, alpha = 1);  convert_element_type_default_6 = convert_element_type_default_7 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 0.5);  convert_element_type_default_5 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_12, alpha = 1);  add_tensor_2 = mul_tensor_12 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_3, input_56);  add_tensor_3 = input_56 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_5, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_57);  permute_default_1 = input_57 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_5, input_63);  view_default_5 = input_63 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_4, input_59);  mm_default_4 = input_59 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(convert_element_type_default_3, convert_element_type_default_8, alpha = 1);  convert_element_type_default_3 = convert_element_type_default_8 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(add_tensor_4, input_64);  input_64 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(input_65, input_66)
        mul_tensor_15 = torch.ops.aten.mul.Tensor(add_tensor_4, mul_tensor_14);  add_tensor_4 = mul_tensor_14 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1], True, dtype = torch.float32);  mul_tensor_15 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(mul_tensor_13, input_65);  input_65 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(mul_tensor_13, input_66);  mul_tensor_13 = input_66 = None
        return (convert_element_type_default_4, mul_tensor_10, convert_element_type_default_9, sum_dim_int_list, mul_tensor_16, mul_tensor_17)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
