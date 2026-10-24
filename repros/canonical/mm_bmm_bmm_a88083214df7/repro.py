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
    def forward(self, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57):
        full_default = torch.ops.aten.full.default(input_37, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_37 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_39);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_40);  view_default = input_40 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_41);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_42);  view_default_1 = input_42 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_43);  clone_default = None
        bmm_default = torch.ops.aten.bmm.default(input_44, view_default_3);  input_44 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_45);  view_default_3 = input_45 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_46);  bmm_default = None
        view_default_5 = torch.ops.aten.view.default(bmm_default_1, input_46);  bmm_default_1 = input_46 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        where_self = torch.ops.aten.where.self(input_47, input_48, input_49);  input_47 = input_48 = input_49 = None
        slice_tensor = torch.ops.aten.slice.Tensor(input_50, 2, 0, 128, 1);  input_50 = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 128, 1);  slice_tensor = None
        where_self_1 = torch.ops.aten.where.self(slice_tensor_1, input_51, input_52);  input_51 = input_52 = None
        add_tensor = torch.ops.aten.add.Tensor(where_self_1, where_self, alpha = 1);  where_self_1 = where_self = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, input_53, alpha = 1);  add_tensor = input_53 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_54);  exp_default = input_54 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, div_tensor);  convert_element_type_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        where_self_2 = torch.ops.aten.where.self(slice_tensor_1, convert_element_type_default_2, full_default);  slice_tensor_1 = convert_element_type_default_2 = full_default = None
        view_default_6 = torch.ops.aten.view.default(where_self_2, input_43);  where_self_2 = input_43 = None
        bmm_default_2 = torch.ops.aten.bmm.default(input_55, view_default_6);  input_55 = None
        bmm_default_3 = torch.ops.aten.bmm.default(view_default_6, input_56);  view_default_6 = input_56 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_1, input_41);  clone_default_1 = input_41 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_39);  view_default_7 = input_39 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(view_default_8, input_57);  view_default_8 = input_57 = None
        return (bmm_default_2, bmm_default_3, permute_default_2, mm_default_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
