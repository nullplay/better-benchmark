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
    def forward(self, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59):
        mul_tensor = torch.ops.aten.mul.Tensor(input_42, input_43);  input_42 = input_43 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_44, torch.float32);  input_44 = None
        exp_default = torch.ops.aten.exp.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [1], True, dtype = None)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_tensor_1, alpha = 1);  convert_element_type_default_1 = mul_tensor_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_3, input_45);  convert_element_type_default_3 = input_45 = None
        add_tensor = torch.ops.aten.add.Tensor(input_46, view_default, alpha = 1);  input_46 = view_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_47);  add_tensor = input_47 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_48);  input_48 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_49);  permute_default = input_49 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_50);  mm_default = input_50 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0, 2]);  convert_element_type_default_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_51, torch.float32);  input_51 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1.1111111111111112);  convert_element_type_default_6 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(permute_default_1, mul_tensor_2);  permute_default_1 = mul_tensor_2 = None
        clone_default = torch.ops.aten.clone.default(mul_tensor_3, memory_format = torch.contiguous_format);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(clone_default, input_52);  input_52 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, 1024)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [2], True, dtype = None)
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_53);  mul_tensor_4 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True, dtype = None);  mul_tensor_6 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(input_53, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_5, sum_dim_int_list_1, alpha = 1);  mul_tensor_5 = sum_dim_int_list_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(clone_default, input_53);  input_53 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1], False, dtype = None);  mul_tensor_8 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(clone_default, [0, 1], False, dtype = None);  clone_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_54, input_55, alpha = 1);  input_54 = input_55 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(input_56, 1.1111111111111112);  input_56 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(add_tensor_1, mul_tensor_9);  add_tensor_1 = mul_tensor_9 = None
        ne_scalar = torch.ops.aten.ne.Scalar(input_57, -1)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(input_58, ne_scalar);  input_58 = ne_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(bitwise_and_tensor, -1);  bitwise_and_tensor = None
        full_default = torch.ops.aten.full.default(input_59, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_59 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_default, [input_57], mul_tensor_10);  full_default = unsqueeze_default = input_57 = mul_tensor_10 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default_5, _unsafe_masked_index_put_accumulate_default, alpha = 1);  convert_element_type_default_5 = _unsafe_masked_index_put_accumulate_default = None
        return (mul_tensor_7, sub_tensor_1, sum_dim_int_list_3, sum_dim_int_list_4, add_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
