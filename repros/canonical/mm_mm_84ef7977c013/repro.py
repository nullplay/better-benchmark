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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_44, torch.bfloat16);  input_44 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        exp_default = torch.ops.aten.exp.default(input_45);  input_45 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [1], True, dtype = None)
        mul_tensor = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_tensor, alpha = 1);  convert_element_type_default_1 = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_46);  convert_element_type_default_2 = input_46 = None
        add_tensor = torch.ops.aten.add.Tensor(input_47, view_default, alpha = 1);  input_47 = view_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_48);  add_tensor = input_48 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default_1, [0, 6, 0, 0], 0);  view_default_1 = None
        mm_default = torch.ops.aten.mm.default(input_49, constant_pad_nd_default);  input_49 = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -6, 1);  mm_default = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(input_50, [0, 0, 0, 6], 0);  input_50 = None
        mm_default_1 = torch.ops.aten.mm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_51);  mm_default_1 = input_51 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(slice_tensor, torch.float32);  slice_tensor = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(convert_element_type_default_4, 0, 0, 128, 1)
        slice_tensor_2 = torch.ops.aten.slice.Tensor(convert_element_type_default_4, 0, 128, 512, 1);  convert_element_type_default_4 = None
        permute_default = torch.ops.aten.permute.default(slice_tensor_1, [1, 0]);  slice_tensor_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_52);  input_52 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 512)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True, dtype = None)
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_53, torch.float32);  input_53 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, input_54, alpha = 1);  convert_element_type_default_5 = input_54 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor_1, input_55);  sub_tensor_1 = input_55 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3);  mul_tensor_1 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True, dtype = None);  mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_3, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list_1, alpha = 1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_5, alpha = 1);  sub_tensor_2 = mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, mul_tensor_3);  mul_tensor_3 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], False, dtype = None);  mul_tensor_6 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_3, [0, 1], False, dtype = None);  convert_element_type_default_3 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_56, input_57, alpha = 1);  input_56 = input_57 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(input_58, input_59, 1, 1, 9223372036854775807, 1);  input_58 = input_59 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, slice_scatter_default, alpha = 1);  add_tensor_1 = slice_scatter_default = None
        ne_scalar = torch.ops.aten.ne.Scalar(input_60, 0)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(input_61, ne_scalar);  input_61 = ne_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(bitwise_and_tensor, -1);  bitwise_and_tensor = None
        full_default = torch.ops.aten.full.default(input_62, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_62 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_default, [input_60], add_tensor_2);  full_default = unsqueeze_default = input_60 = add_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(permute_default, _unsafe_masked_index_put_accumulate_default, alpha = 1);  permute_default = _unsafe_masked_index_put_accumulate_default = None
        return (slice_tensor_2, sub_tensor_3, sum_dim_int_list_3, sum_dim_int_list_4, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
