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
    def forward(self, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51):
        add_tensor = torch.ops.aten.add.Tensor(input_34, input_35, alpha = 1);  input_34 = input_35 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_36);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_37);  input_37 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_38);  input_38 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_39);  mm_default = input_39 = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default, input_40);  permute_default = input_40 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default, input_41);  view_default = input_41 = None
        view_default_2 = torch.ops.aten.view.default(mm_default_3, input_42);  mm_default_3 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_43);  view_default_2 = input_43 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_1, input_44);  permute_default_1 = input_44 = None
        bmm_default = torch.ops.aten.bmm.default(input_45, view_default_4);  input_45 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_4, input_46);  view_default_4 = input_46 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default, input_47);  bmm_default = input_47 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default_1, input_48);  bmm_default_1 = input_48 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, input_49);  convert_element_type_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(input_49);  input_49 = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_2, 16.0);  convert_element_type_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_42);  clone_default = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_36);  view_default_7 = input_36 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_3, input_50);  permute_default_3 = input_50 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_8, input_51);  view_default_8 = input_51 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_5, input_42);  mm_default_5 = input_42 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        return (mm_default_1, view_default_1, mm_default_2, div_tensor, convert_element_type_default_3, convert_element_type_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
