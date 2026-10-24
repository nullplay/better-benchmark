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
    def forward(self, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        permute_default = torch.ops.aten.permute.default(input_34, [1, 0, 2, 3]);  input_34 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default = torch.ops.aten.view.default(permute_default_1, input_35);  permute_default_1 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_36);  view_default = None
        as_strided_default = torch.ops.aten.as_strided.default(view_default_1, input_37, input_38, None);  view_default_1 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default = torch.ops.aten.clone.default(unsqueeze_default, memory_format = torch.contiguous_format);  unsqueeze_default = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_39);  clone_default = input_39 = None
        view_default_3 = torch.ops.aten.view.default(input_40, input_35);  input_40 = input_35 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_36);  view_default_3 = input_36 = None
        as_strided_default_1 = torch.ops.aten.as_strided.default(view_default_4, input_37, input_38, None);  view_default_4 = input_37 = input_38 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(as_strided_default_1, 4);  as_strided_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(unsqueeze_default_1, [0, 1, 4, 2, 3]);  unsqueeze_default_1 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 4, 3, 2]);  permute_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_41);  clone_default_1 = input_41 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_2, view_default_5);  view_default_2 = view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default, input_42);  bmm_default = input_42 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_6, [0, 1, 2, 4, 3]);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(permute_default_4, input_43);  permute_default_4 = input_43 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default_7, [0, 0, 0, 1], 0.0);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(constant_pad_nd_default, input_44);  constant_pad_nd_default = input_44 = None
        slice_tensor = torch.ops.aten.slice.Tensor(view_default_8, 2, 0, 256, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(slice_tensor, 3, 0, 257, 1);  slice_tensor = None
        copy_default = torch.ops.aten.copy.default(input_45, slice_tensor_1, False);  input_45 = slice_tensor_1 = None
        select_int = torch.ops.aten.select.int(view_default_8, 1, -1)
        slice_tensor_2 = torch.ops.aten.slice.Tensor(select_int, 1, 256, 9223372036854775807, 1);  select_int = None
        slice_tensor_3 = torch.ops.aten.slice.Tensor(slice_tensor_2, 2, 0, 257, 1);  slice_tensor_2 = None
        slice_tensor_4 = torch.ops.aten.slice.Tensor(view_default_8, 2, -257, -1, 1)
        slice_tensor_5 = torch.ops.aten.slice.Tensor(slice_tensor_4, 3, 257, 9223372036854775807, 1);  slice_tensor_4 = None
        copy_default_1 = torch.ops.aten.copy.default(input_46, slice_tensor_5, False);  input_46 = slice_tensor_5 = None
        select_int_1 = torch.ops.aten.select.int(view_default_8, 1, 0);  view_default_8 = None
        slice_tensor_6 = torch.ops.aten.slice.Tensor(select_int_1, 1, 0, 255, 1);  select_int_1 = None
        slice_tensor_7 = torch.ops.aten.slice.Tensor(slice_tensor_6, 2, -255, 9223372036854775807, 1);  slice_tensor_6 = None
        return (copy_default, slice_tensor_3, copy_default_1, slice_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
