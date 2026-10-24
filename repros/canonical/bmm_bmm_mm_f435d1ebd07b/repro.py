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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42):
        clone_default = torch.ops.aten.clone.default(input_30, memory_format = torch.contiguous_format);  input_30 = None
        view_default = torch.ops.aten.view.default(clone_default, input_31);  clone_default = input_31 = None
        view_default_1 = torch.ops.aten.view.default(input_32, input_33);  input_32 = input_33 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 2, 1]);  view_default_1 = None
        bmm_default = torch.ops.aten.bmm.default(permute_default, view_default);  permute_default = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default, input_34);  view_default = input_34 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default, input_35);  bmm_default = input_35 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default_1, input_36);  clone_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_37);  view_default_3 = None
        mm_default = torch.ops.aten.mm.default(view_default_4, input_38);  input_38 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_39);  permute_default_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(view_default_4, [0], True, dtype = torch.float32);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(sum_dim_int_list, input_40);  sum_dim_int_list = input_40 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_5, torch.bfloat16);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(mm_default, input_36);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        clone_default_2 = torch.ops.aten.clone.default(input_41, memory_format = torch.contiguous_format);  input_41 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_2, input_37);  clone_default_2 = input_37 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_7, input_42);  input_42 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [1, 0]);  view_default_7 = None
        mm_default_3 = torch.ops.aten.mm.default(permute_default_3, input_39);  permute_default_3 = input_39 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_2, input_36);  mm_default_2 = input_36 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default_1, convert_element_type_default_4, alpha = 1);  convert_element_type_default_1 = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        return (bmm_default_1, convert_element_type_default_2, convert_element_type_default_3, add_tensor, convert_element_type_default_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
