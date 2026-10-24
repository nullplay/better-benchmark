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
    def forward(self, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        mul_tensor = torch.ops.aten.mul.Tensor(input_36, input_37);  input_36 = input_37 = None
        view_default = torch.ops.aten.view.default(mul_tensor, input_38);  mul_tensor = None
        mm_default = torch.ops.aten.mm.default(view_default, input_39);  view_default = input_39 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_40);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_41);  view_default_1 = input_41 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_42);  clone_default = input_42 = None
        bmm_default = torch.ops.aten.bmm.default(input_43, view_default_3);  input_43 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_44);  view_default_3 = input_44 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(bmm_default_1, torch.float32);  bmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default, input_45);  convert_element_type_default = input_45 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_46, torch.float32);  input_46 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_4, mul_tensor_1);  view_default_4 = mul_tensor_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_47, torch.float32);  input_47 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_2, input_48, alpha = 1);  convert_element_type_default_2 = input_48 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_49);  exp_default = input_49 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, div_tensor);  mul_tensor_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_3);  neg_default = sum_dim_int_list = mul_tensor_3 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        full_default = torch.ops.aten.full.default(input_50, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_50 = None
        full_default_1 = torch.ops.aten.full.default(input_51, False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_51 = None
        where_self = torch.ops.aten.where.self(full_default_1, full_default, convert_element_type_default_3);  full_default_1 = full_default = convert_element_type_default_3 = None
        view_default_5 = torch.ops.aten.view.default(where_self, input_52);  where_self = input_52 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default, input_53);  bmm_default = input_53 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_1, input_40);  clone_default_1 = input_40 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_38);  view_default_7 = input_38 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_8, input_54);  input_54 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_8, [0], True, dtype = torch.float32);  view_default_8 = None
        return (view_default_5, mm_default_1, permute_default_2, sum_dim_int_list_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
