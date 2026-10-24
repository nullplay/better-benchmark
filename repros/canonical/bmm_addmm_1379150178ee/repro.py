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
    def forward(self, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31):
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(input_20, [0, 7, 0, 7, 0, 0], 0);  input_20 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(input_21, [0, 0, 0, 7, 0, 0], 0);  input_21 = None
        bmm_default = torch.ops.aten.bmm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        slice_tensor = torch.ops.aten.slice.Tensor(bmm_default, 1, 0, -7, 1);  bmm_default = None
        view_default = torch.ops.aten.view.default(slice_tensor, input_22);  slice_tensor = input_22 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 2, 1, 3]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_23);  clone_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_24);  view_default_1 = input_24 = None
        permute_default_1 = torch.ops.aten.permute.default(input_25, [1, 0]);  input_25 = None
        addmm_default = torch.ops.aten.addmm.default(input_26, view_default_2, permute_default_1, beta = 1, alpha = 1);  input_26 = view_default_2 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default, input_23);  addmm_default = input_23 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_27);  view_default_3 = input_27 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_28);  view_default_4 = input_28 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 1, 3, 2, 4, 5]);  view_default_5 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_29);  clone_default_1 = input_29 = None
        fmod_scalar = torch.ops.aten.fmod.Scalar(input_30, 14);  input_30 = None
        index_tensor = torch.ops.aten.index.Tensor(view_default_6, [None, fmod_scalar]);  view_default_6 = fmod_scalar = None
        index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, input_31]);  index_tensor = input_31 = None
        return (index_tensor_1,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
