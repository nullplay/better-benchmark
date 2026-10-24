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
    def forward(self, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95):
        div_tensor = torch.ops.aten.div.Tensor(input_67, input_68);  input_67 = input_68 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_69, 1);  input_69 = None
        ne_scalar = torch.ops.aten.ne.Scalar(unsqueeze_default, -100)
        where_self = torch.ops.aten.where.self(ne_scalar, unsqueeze_default, input_70);  unsqueeze_default = input_70 = None
        view_default = torch.ops.aten.view.default(input_71, input_72);  input_71 = input_72 = None
        expand_default = torch.ops.aten.expand.default(where_self, input_73, implicit = False);  where_self = None
        eq_tensor = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = None)
        scalar_tensor_default_1 = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = None)
        where_self_1 = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default = torch.ops.aten.full.default(input_74, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_74 = None
        where_self_2 = torch.ops.aten.where.self(ne_scalar, div_tensor, full_default);  ne_scalar = div_tensor = full_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(where_self_1, where_self_2);  where_self_1 = where_self_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_75, input_76, alpha = 1);  input_75 = input_76 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, input_77, alpha = 1);  sub_tensor = input_77 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1);  sub_tensor_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True, dtype = None)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor, mul_tensor_1, alpha = 1);  mul_tensor = mul_tensor_1 = None
        view_default_1 = torch.ops.aten.view.default(sub_tensor_2, input_78);  sub_tensor_2 = input_78 = None
        add_tensor = torch.ops.aten.add.Tensor(input_79, view_default_1, alpha = 1);  input_79 = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor, input_73);  add_tensor = input_73 = None
        permute_default = torch.ops.aten.permute.default(input_80, [1, 0]);  input_80 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mm_default = torch.ops.aten.mm.default(view_default_2, permute_default_1);  permute_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_81);  permute_default_2 = input_81 = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_82);  mm_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_3, input_83);  input_83 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 768)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None)
        view_default_4 = torch.ops.aten.view.default(input_84, input_82);  input_84 = input_82 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default_4, 0.5)
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_4, input_85, alpha = 1);  view_default_4 = input_85 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_1, 0.7978845608028654);  add_tensor_1 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_4, add_tensor_2)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, input_86, alpha = 1);  mul_tensor_6 = input_86 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_87);  sub_tensor_3 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_2, mul_tensor_7);  mul_tensor_2 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True, dtype = None);  mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_7, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(mul_tensor_3, sum_dim_int_list_1, alpha = 1);  mul_tensor_3 = sum_dim_int_list_1 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(sub_tensor_4, mul_tensor_9, alpha = 1);  sub_tensor_4 = mul_tensor_9 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(input_87, 768);  input_87 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(div_tensor_1, sub_tensor_5);  div_tensor_1 = sub_tensor_5 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(view_default_3, mul_tensor_7);  mul_tensor_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1], False, dtype = None);  mul_tensor_11 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(view_default_3, [0, 1], False, dtype = None);  view_default_3 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(mul_tensor_10, mul_tensor_4);  mul_tensor_4 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(mul_tensor_10, add_tensor_2);  mul_tensor_10 = add_tensor_2 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(mul_tensor_12, input_88);  mul_tensor_12 = input_88 = None
        mul_tensor_15 = torch.ops.aten.mul.Tensor(mul_tensor_13, 0.5);  mul_tensor_13 = None
        mul_tensor_16 = torch.ops.aten.mul.Tensor(input_89, input_90);  input_89 = input_90 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(input_91, input_92, alpha = 1);  input_91 = input_92 = None
        sub_tensor_7 = torch.ops.aten.sub.Tensor(sub_tensor_6, mul_tensor_16, alpha = 1);  sub_tensor_6 = mul_tensor_16 = None
        mul_tensor_17 = torch.ops.aten.mul.Tensor(input_93, sub_tensor_7);  input_93 = sub_tensor_7 = None
        ge_scalar = torch.ops.aten.ge.Scalar(input_94, 0)
        lt_scalar = torch.ops.aten.lt.Scalar(input_94, 32000)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(ge_scalar, lt_scalar);  ge_scalar = lt_scalar = None
        ne_scalar_1 = torch.ops.aten.ne.Scalar(input_94, 3)
        bitwise_and_tensor_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, ne_scalar_1);  bitwise_and_tensor = ne_scalar_1 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(bitwise_and_tensor_1, -1);  bitwise_and_tensor_1 = None
        full_default_1 = torch.ops.aten.full.default(input_95, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_95 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_1, unsqueeze_default_1, [input_94], mul_tensor_17);  full_default_1 = unsqueeze_default_1 = input_94 = mul_tensor_17 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mm_default_1, _unsafe_masked_index_put_accumulate_default, alpha = 1);  mm_default_1 = _unsafe_masked_index_put_accumulate_default = None
        return (sum_dim_int_list_3, sum_dim_int_list_4, mul_tensor_14, mul_tensor_15, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
