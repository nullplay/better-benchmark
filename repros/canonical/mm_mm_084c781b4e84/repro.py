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
    def forward(self, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80):
        mul_tensor = torch.ops.aten.mul.Tensor(input_54, input_55);  input_54 = input_55 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_56, torch.float32);  input_56 = None
        exp_default = torch.ops.aten.exp.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [1], True, dtype = None)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_tensor_1, alpha = 1);  convert_element_type_default_1 = mul_tensor_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_3, input_57);  convert_element_type_default_3 = input_57 = None
        add_tensor = torch.ops.aten.add.Tensor(input_58, view_default, alpha = 1);  input_58 = view_default = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_59);  add_tensor = input_59 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_60);  permute_default = input_60 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, input_61);  view_default_1 = input_61 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_62);  mm_default_1 = input_62 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_63, torch.float32);  input_63 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, 1.1111111111111112);  convert_element_type_default_6 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, mul_tensor_2);  convert_element_type_default_4 = mul_tensor_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_64);  input_64 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(input_65, input_66)
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_3, mul_tensor_5);  mul_tensor_3 = mul_tensor_5 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], True, dtype = torch.float32);  mul_tensor_6 = None
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list_1, input_67);  sum_dim_int_list_1 = input_67 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_65)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_66);  mul_tensor_4 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True, dtype = torch.float32);  mul_tensor_7 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_66, 3);  input_66 = None
        mul_scalar = torch.ops.aten.mul.Scalar(sum_dim_int_list_2, -0.5);  sum_dim_int_list_2 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        div_scalar = torch.ops.aten.div.Scalar(input_68, 512);  input_68 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(input_65, 1.0);  input_65 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar_1);  div_scalar = mul_scalar_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_8, mul_tensor_10, alpha = 1);  mul_tensor_8 = mul_tensor_10 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16)
        mul_tensor_11 = torch.ops.aten.mul.Tensor(convert_element_type_default_7, input_69);  convert_element_type_default_7 = input_69 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(input_70, input_71);  input_70 = input_71 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_12, alpha = 1);  add_tensor_1 = mul_tensor_12 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, input_72, alpha = 1);  add_tensor_2 = input_72 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(input_73, input_74, alpha = 1);  input_73 = input_74 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_75, 1.1111111111111112);  input_75 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(add_tensor_4, mul_tensor_13);  add_tensor_4 = mul_tensor_13 = None
        ne_scalar = torch.ops.aten.ne.Scalar(input_76, -1)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(input_77, ne_scalar);  input_77 = ne_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(bitwise_and_tensor, -1);  bitwise_and_tensor = None
        full_default = torch.ops.aten.full.default(input_78, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_78 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_default, [input_76], mul_tensor_14);  mul_tensor_14 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(convert_element_type_default_5, _unsafe_masked_index_put_accumulate_default, alpha = 1);  convert_element_type_default_5 = _unsafe_masked_index_put_accumulate_default = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(input_79, input_80);  input_79 = input_80 = None
        _unsafe_masked_index_put_accumulate_default_1 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default, unsqueeze_default, [input_76], mul_tensor_15);  full_default = unsqueeze_default = input_76 = mul_tensor_15 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, _unsafe_masked_index_put_accumulate_default_1, alpha = 1);  add_tensor_5 = _unsafe_masked_index_put_accumulate_default_1 = None
        return (view_default_3, mul_tensor_9, mul_tensor_11, add_tensor_3, add_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
