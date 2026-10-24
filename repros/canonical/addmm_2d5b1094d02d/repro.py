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
    def forward(self, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40):
        add_tensor = torch.ops.aten.add.Tensor(input_28, input_29, alpha = 1);  input_28 = input_29 = None
        view_default = torch.ops.aten.view.default(add_tensor, input_30);  add_tensor = input_30 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2, 4, 5]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_31);  clone_default = input_31 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_32);  view_default_1 = input_32 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_34, torch.bfloat16);  input_34 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.bfloat16);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_2, input_35);  convert_element_type_default_2 = input_35 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default = view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default, input_36);  addmm_default = input_36 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_37);  view_default_4 = input_37 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [2, 0, 3, 1, 4]);  view_default_5 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default_2, 0);  permute_default_2 = None
        getitem = unbind_int[0]
        getitem_1 = unbind_int[1]
        getitem_2 = unbind_int[2];  unbind_int = None
        mul_tensor = torch.ops.aten.mul.Tensor(getitem, 0.1767766952966369);  getitem = None
        permute_default_3 = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default = torch.ops.aten.expand.default(mul_tensor, input_38, implicit = False);  mul_tensor = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_3, input_39, implicit = False);  permute_default_3 = input_39 = None
        clone_default_2 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        expand_default_2 = torch.ops.aten.expand.default(getitem_2, input_38, implicit = False);  getitem_2 = input_38 = None
        clone_default_3 = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_3, input_40);  clone_default_3 = input_40 = None
        return (clone_default_1, clone_default_2, view_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
