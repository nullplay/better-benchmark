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
    def forward(self, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83):
        where_self = torch.ops.aten.where.self(input_52, input_53, input_54);  input_52 = input_53 = input_54 = None
        where_self_1 = torch.ops.aten.where.self(input_55, input_56, input_57);  input_55 = input_56 = input_57 = None
        mul_tensor = torch.ops.aten.mul.Tensor(where_self, where_self_1);  where_self = where_self_1 = None
        exp_default = torch.ops.aten.exp.default(input_58);  input_58 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True, dtype = None)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1, alpha = 1);  mul_tensor = mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(sub_tensor, input_59);  sub_tensor = input_59 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        add_tensor = torch.ops.aten.add.Tensor(input_60, convert_element_type_default, alpha = 1);  input_60 = convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_61);  add_tensor = input_61 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_62);  input_62 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_63);  permute_default = input_63 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_64);  mm_default = input_64 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, input_65);  input_65 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1024)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_66, input_67, alpha = 1);  input_66 = input_67 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_1, input_68);  sub_tensor_1 = input_68 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_4);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None);  mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_1, alpha = 1);  mul_tensor_3 = sum_dim_int_list_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, mul_tensor_4);  mul_tensor_4 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1], False, dtype = None);  mul_tensor_7 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [0, 1], False, dtype = None);  convert_element_type_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_69, input_70, alpha = 1);  input_69 = input_70 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_71, input_72);  input_71 = input_72 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_8, alpha = 1);  add_tensor_1 = mul_tensor_8 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(input_73, input_74, alpha = 1);  input_73 = input_74 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(input_75, sub_tensor_3);  input_75 = sub_tensor_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_9, alpha = 1);  add_tensor_2 = mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_76, input_77);  input_76 = input_77 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(input_78, input_79, alpha = 1);  input_78 = input_79 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_10, alpha = 1);  sub_tensor_4 = mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_80, sub_tensor_5);  input_80 = sub_tensor_5 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, mul_tensor_11, alpha = 1);  add_tensor_3 = mul_tensor_11 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_81, torch.float32);  input_81 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, 1.1111111111111112);  convert_element_type_default_3 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(add_tensor_4, mul_tensor_12);  add_tensor_4 = mul_tensor_12 = None
        ge_scalar = torch.ops.aten.ge.Scalar(input_82, 0)
        lt_scalar = torch.ops.aten.lt.Scalar(input_82, 29056)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(ge_scalar, lt_scalar);  ge_scalar = lt_scalar = None
        ne_scalar = torch.ops.aten.ne.Scalar(input_82, 0)
        bitwise_and_tensor_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, ne_scalar);  bitwise_and_tensor = ne_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(bitwise_and_tensor_1, -1);  bitwise_and_tensor_1 = None
        full_default = torch.ops.aten.full.default(input_83, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_83 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_default, [input_82], mul_tensor_13);  full_default = unsqueeze_default = input_82 = mul_tensor_13 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(convert_element_type_default_2, _unsafe_masked_index_put_accumulate_default, alpha = 1);  convert_element_type_default_2 = _unsafe_masked_index_put_accumulate_default = None
        return (mul_tensor_6, sub_tensor_2, sum_dim_int_list_3, sum_dim_int_list_4, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
