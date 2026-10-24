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
    def forward(self, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67):
        mul_tensor = torch.ops.aten.mul.Tensor(input_48, input_49);  input_48 = input_49 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_50, torch.float32);  input_50 = None
        exp_default = torch.ops.aten.exp.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [1], True, dtype = None)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_tensor_1, alpha = 1);  convert_element_type_default_1 = mul_tensor_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_3, input_51);  convert_element_type_default_3 = input_51 = None
        add_tensor = torch.ops.aten.add.Tensor(input_52, view_default, alpha = 1);  input_52 = view_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_53);  add_tensor = input_53 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_54);  input_54 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_55);  permute_default = input_55 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_56);  mm_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_57);  input_57 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 128)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None)
        view_default_3 = torch.ops.aten.view.default(input_58, input_56);  input_58 = input_56 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default_3, 0.5);  view_default_3 = None
        tanh_default = torch.ops.aten.tanh.default(input_59);  input_59 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_1)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_5, input_60, alpha = 1);  mul_tensor_5 = input_60 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, input_61);  sub_tensor_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_6);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True, dtype = None);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_6, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_1, alpha = 1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_8, alpha = 1);  sub_tensor_2 = mul_tensor_8 = None
        div_tensor = torch.ops.aten.div.Tensor(input_61, 128);  input_61 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_3);  div_tensor = sub_tensor_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, mul_tensor_6);  mul_tensor_6 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1], False, dtype = None);  mul_tensor_10 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 1], False, dtype = None);  convert_element_type_default_4 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_9, mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_9, add_tensor_1);  mul_tensor_9 = add_tensor_1 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(input_62, input_63, alpha = 1);  input_62 = input_63 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_64, sub_tensor_4);  input_64 = sub_tensor_4 = None
        ne_scalar = torch.ops.aten.ne.Scalar(input_65, 0)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(input_66, ne_scalar);  input_66 = ne_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(bitwise_and_tensor, -1);  bitwise_and_tensor = None
        full_default = torch.ops.aten.full.default(input_67, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_67 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_default, [input_65], mul_tensor_13);  full_default = unsqueeze_default = input_65 = mul_tensor_13 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default_5, _unsafe_masked_index_put_accumulate_default, alpha = 1);  convert_element_type_default_5 = _unsafe_masked_index_put_accumulate_default = None
        return (sum_dim_int_list_3, sum_dim_int_list_4, mul_tensor_11, mul_tensor_12, add_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
