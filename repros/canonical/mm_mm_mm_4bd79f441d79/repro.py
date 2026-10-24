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
    def forward(self, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65):
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_46, [-1], True, dtype = None)
        exp_default = torch.ops.aten.exp.default(input_47);  input_47 = None
        mul_tensor = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_46, mul_tensor, alpha = 1);  input_46 = mul_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_48);  convert_element_type_default = input_48 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default, [0, 3, 0, 0], 0)
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(input_49, [0, 0, 0, 3], 0);  input_49 = None
        mm_default = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 0, 0, 3], 0);  permute_default = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default_2, input_50);  constant_pad_nd_default_2 = input_50 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default_1, 0, 0, -3, 1);  mm_default_1 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_51);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_52, input_53, alpha = 1);  input_52 = input_53 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        mm_default_2 = torch.ops.aten.mm.default(convert_element_type_default_3, input_54);  convert_element_type_default_3 = input_54 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        full_default = torch.ops.aten.full.default(input_51, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_51 = None
        select_scatter_default = torch.ops.aten.select_scatter.default(full_default, convert_element_type_default_4, 1, 0);  full_default = convert_element_type_default_4 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default_1, select_scatter_default, alpha = 1);  convert_element_type_default_1 = select_scatter_default = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_55, torch.float32);  input_55 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, 1.1111111111111112);  convert_element_type_default_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_1);  add_tensor = mul_tensor_1 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16)
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_56, torch.bfloat16);  input_56 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, 1.1111111111111112);  convert_element_type_default_7 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, mul_tensor_3);  convert_element_type_default_6 = mul_tensor_3 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_4, input_57);  mul_tensor_4 = input_57 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_2, input_58);  input_58 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0])
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_2, [0], True, dtype = torch.float32);  view_default_2 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_59, torch.float32);  input_59 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_60, 1e-06, alpha = 1);  input_60 = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_8, add_tensor_1);  convert_element_type_default_8 = add_tensor_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, input_61);  div_tensor = input_61 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_5, alpha = 1);  mul_tensor_2 = mul_tensor_5 = None
        mul_scalar = torch.ops.aten.mul.Scalar(input_62, 0.002607561929595828);  input_62 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_scalar, input_63);  mul_scalar = input_63 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_6, alpha = 1);  add_tensor_2 = mul_tensor_6 = None
        div_scalar = torch.ops.aten.div.Scalar(input_64, 768);  input_64 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, div_scalar, alpha = 1);  add_tensor_3 = div_scalar = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16)
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, input_65, alpha = 1);  add_tensor_4 = input_65 = None
        return (convert_element_type_default_2, mm_default_3, permute_default_1, sum_dim_int_list_1, convert_element_type_default_9, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
