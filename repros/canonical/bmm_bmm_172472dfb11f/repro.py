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
    def forward(self, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48):
        view_default = torch.ops.aten.view.default(input_28, input_29);  input_28 = input_29 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0, 2, 3]);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_30);  clone_default = input_30 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_31);  view_default_1 = input_31 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_2, [0, 1, 2, 4, 3]);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(permute_default_2, input_32);  permute_default_2 = input_32 = None
        bmm_default = torch.ops.aten.bmm.default(input_33, view_default_3);  input_33 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_34);  view_default_3 = input_34 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_35);  bmm_default = input_35 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default_1, input_36);  bmm_default_1 = input_36 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default_4, 4);  view_default_4 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(view_default_5, 4);  view_default_5 = None
        full_default = torch.ops.aten.full.default(input_37, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_37 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(full_default, squeeze_dim_1, 3, 0, -1, 1);  full_default = squeeze_dim_1 = None
        view_default_6 = torch.ops.aten.view.default(slice_scatter_default, input_38);  slice_scatter_default = input_38 = None
        full_default_1 = torch.ops.aten.full.default(input_39, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_39 = None
        slice_scatter_default_1 = torch.ops.aten.slice_scatter.default(full_default_1, view_default_6, 2, 0, -256, 1);  full_default_1 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(slice_scatter_default_1, input_40);  slice_scatter_default_1 = input_40 = None
        full_default_2 = torch.ops.aten.full.default(input_41, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_41 = None
        view_default_8 = torch.ops.aten.view.default(squeeze_dim, input_42);  squeeze_dim = input_42 = None
        view_default_9 = torch.ops.aten.view.default(input_43, input_44);  input_43 = input_44 = None
        index_add_default = torch.ops.aten.index_add.default(full_default_2, 0, view_default_9, view_default_8, alpha = 1);  full_default_2 = view_default_9 = view_default_8 = None
        as_strided_default = torch.ops.aten.as_strided.default(index_add_default, input_45, input_46, input_47);  index_add_default = input_45 = input_46 = input_47 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(as_strided_default, [0, 0, -256, -256], 0);  as_strided_default = None
        view_default_10 = torch.ops.aten.view.default(constant_pad_nd_default, input_48);  constant_pad_nd_default = input_48 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        return (view_default_7, permute_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
