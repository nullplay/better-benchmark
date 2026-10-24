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
    def forward(self, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_37, torch.float32);  input_37 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_38, 1.1111111111111112);  input_38 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        exp_default = torch.ops.aten.exp.default(input_39);  input_39 = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_40);  exp_default = input_40 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor);  mul_tensor_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        full_default = torch.ops.aten.full.default(input_41, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_41 = None
        eq_scalar = torch.ops.aten.eq.Scalar(input_42, 0);  input_42 = None
        where_self = torch.ops.aten.where.self(eq_scalar, full_default, convert_element_type_default_1);  eq_scalar = full_default = convert_element_type_default_1 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(where_self, 8.0);  where_self = None
        view_default = torch.ops.aten.view.default(div_tensor_1, input_43);  div_tensor_1 = input_43 = None
        bmm_default = torch.ops.aten.bmm.default(input_44, view_default);  input_44 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default, input_45);  view_default = input_45 = None
        view_default_1 = torch.ops.aten.view.default(bmm_default, input_46);  bmm_default = input_46 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default_1, input_47);  bmm_default_1 = input_47 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 2]);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(permute_default_1, input_48);  permute_default_1 = None
        clone_default = torch.ops.aten.clone.default(view_default_3, memory_format = torch.contiguous_format);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_49);  clone_default = None
        mm_default = torch.ops.aten.mm.default(view_default_4, input_50);  input_50 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_51);  permute_default_2 = input_51 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True, dtype = torch.float32);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(sum_dim_int_list_1, input_52);  sum_dim_int_list_1 = input_52 = None
        view_default_6 = torch.ops.aten.view.default(mm_default, input_48);  mm_default = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_1, input_48);  clone_default_1 = input_48 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_49);  view_default_7 = input_49 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_8, input_53);  input_53 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_8, [0], True, dtype = torch.float32);  view_default_8 = None
        return (mm_default_1, view_default_5, view_default_6, mm_default_2, permute_default_4, sum_dim_int_list_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
