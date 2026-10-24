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
    def forward(self, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48):
        view_default = torch.ops.aten.view.default(input_37, input_38);  input_37 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 3, 4, 2]);  view_default = None
        view_default_1 = torch.ops.aten.view.default(permute_default, input_39);  permute_default = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_1, input_40, alpha = 1);  view_default_1 = input_40 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(add_tensor, 4);  add_tensor = None
        permute_default_1 = torch.ops.aten.permute.default(unsqueeze_default, [1, 2, 0, 4, 3]);  unsqueeze_default = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default_1, [0, 1, 2, 4, 3]);  permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(permute_default_2, input_41);  permute_default_2 = input_41 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_42, 0);  input_42 = None
        view_default_3 = torch.ops.aten.view.default(unsqueeze_default_1, input_38);  unsqueeze_default_1 = input_38 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_3, [0, 1, 3, 4, 2]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_3, input_39);  permute_default_3 = input_39 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(view_default_4, 4);  view_default_4 = None
        permute_default_4 = torch.ops.aten.permute.default(unsqueeze_default_2, [1, 2, 4, 0, 3]);  unsqueeze_default_2 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 1, 4, 3, 2]);  permute_default_4 = None
        view_default_5 = torch.ops.aten.view.default(permute_default_5, input_43);  permute_default_5 = input_43 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_2, view_default_5);  view_default_2 = view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default, input_44);  bmm_default = input_44 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_6, [0, 1, 2, 4, 3]);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(permute_default_6, input_45);  permute_default_6 = input_45 = None
        view_default_8 = torch.ops.aten.view.default(input_46, input_47);  input_46 = input_47 = None
        slice_tensor = torch.ops.aten.slice.Tensor(view_default_8, 2, 1, 9223372036854775807, 1);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(slice_tensor, input_48);  slice_tensor = input_48 = None
        iota_default = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index_tensor = torch.ops.aten.index.Tensor(view_default_9, [None, None, None, iota_default]);  view_default_9 = iota_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_7, index_tensor, alpha = 1);  view_default_7 = index_tensor = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, 0, alpha = 1);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_2, 0.125)
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        eq_tensor = torch.ops.aten.eq.Tensor(convert_element_type_default, convert_element_type_default)
        abs_default = torch.ops.aten.abs.default(convert_element_type_default)
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32);  add_tensor_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor_1, [3], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default, alpha = 1);  mul_tensor_1 = amax_default = None
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default, [3], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1, alpha = 1);  convert_element_type_default = amax_default_1 = None
        return (eq_tensor, abs_default, sub_tensor, sub_tensor_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
