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
    def forward(self, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38):
        view_default = torch.ops.aten.view.default(input_25, input_26);  input_25 = input_26 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default, [0, 257], 0.0);  view_default = None
        view_default_1 = torch.ops.aten.view.default(constant_pad_nd_default, input_27);  constant_pad_nd_default = input_27 = None
        slice_tensor = torch.ops.aten.slice.Tensor(view_default_1, 2, 0, -256, 1);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(slice_tensor, input_28);  slice_tensor = input_28 = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(view_default_2, 3, 0, -1, 1);  view_default_2 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(slice_tensor_1, 4);  slice_tensor_1 = None
        view_default_3 = torch.ops.aten.view.default(unsqueeze_default, input_29);  unsqueeze_default = input_29 = None
        permute_default = torch.ops.aten.permute.default(input_30, [1, 0, 2, 3]);  input_30 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_4 = torch.ops.aten.view.default(permute_default_1, input_31);  permute_default_1 = input_31 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(view_default_4, [0, 0, 256, 256], -1.0);  view_default_4 = None
        as_strided_default = torch.ops.aten.as_strided.default(constant_pad_nd_default_1, input_32, input_33, None);  constant_pad_nd_default_1 = input_32 = input_33 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(as_strided_default, 4);  as_strided_default = None
        clone_default = torch.ops.aten.clone.default(unsqueeze_default_1, memory_format = torch.contiguous_format);  unsqueeze_default_1 = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_34);  clone_default = input_34 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_3, view_default_5);  view_default_3 = view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default, input_35);  bmm_default = input_35 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [0, 1, 2, 4, 3]);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(permute_default_2, input_36);  permute_default_2 = input_36 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_37);  view_default_7 = input_37 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_3, [1, 0, 2, 3]);  permute_default_3 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_1, input_38);  clone_default_1 = input_38 = None
        return (view_default_9,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
