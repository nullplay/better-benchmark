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
    def forward(self, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_32, torch.bfloat16);  input_32 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_33);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_34);  view_default = input_34 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_35);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_36);  view_default_1 = input_36 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_37);  clone_default = input_37 = None
        bmm_default = torch.ops.aten.bmm.default(input_38, view_default_3);  input_38 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_39);  view_default_3 = input_39 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_40);  bmm_default = None
        view_default_5 = torch.ops.aten.view.default(bmm_default_1, input_40);  bmm_default_1 = input_40 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_41, input_42, alpha = 1);  input_41 = input_42 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_43);  exp_default = input_43 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, div_tensor);  convert_element_type_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        where_self = torch.ops.aten.where.self(input_44, convert_element_type_default_2, input_45);  input_44 = convert_element_type_default_2 = input_45 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_35);  clone_default_1 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_33);  view_default_6 = input_33 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_46);  permute_default_2 = input_46 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_7, input_47);  view_default_7 = input_47 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_2, input_35);  mm_default_2 = input_35 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        return (where_self, convert_element_type_default_3, convert_element_type_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
