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
    def forward(self, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        add_tensor = torch.ops.aten.add.Tensor(input_25, input_26, alpha = 1);  input_25 = input_26 = None
        view_default = torch.ops.aten.view.default(add_tensor, input_27);  add_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_28, torch.bfloat16);  input_28 = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default, 0.8999999985098839);  convert_element_type_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default, div_tensor);  view_default = div_tensor = None
        view_default_1 = torch.ops.aten.view.default(mul_tensor, input_29);  mul_tensor = input_29 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 1, 3, 2, 4, 5]);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(permute_default, input_27);  permute_default = input_27 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_30);  view_default_2 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_31);  view_default_3 = input_31 = None
        mm_default = torch.ops.aten.mm.default(view_default_4, input_32);  input_32 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_1, input_33);  permute_default_1 = input_33 = None
        view_default_5 = torch.ops.aten.view.default(mm_default, input_30);  mm_default = input_30 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_34);  view_default_5 = input_34 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_35);  clone_default = input_35 = None
        bmm_default = torch.ops.aten.bmm.default(input_36, view_default_7);  input_36 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_7, input_37);  view_default_7 = input_37 = None
        view_default_8 = torch.ops.aten.view.default(bmm_default, input_38);  bmm_default = input_38 = None
        view_default_9 = torch.ops.aten.view.default(bmm_default_1, input_39);  bmm_default_1 = input_39 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        cat_default = torch.ops.aten.cat.default([input_40, input_41, view_default_8], 0);  input_40 = input_41 = view_default_8 = None
        return (convert_element_type_default_1, convert_element_type_default_2, cat_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
