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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50):
        full_default = torch.ops.aten.full.default(input_30, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_30 = None
        add_tensor = torch.ops.aten.add.Tensor(input_31, input_32, alpha = 1);  input_31 = input_32 = None
        select_scatter_default = torch.ops.aten.select_scatter.default(full_default, input_33, 1, -1);  input_33 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, select_scatter_default, alpha = 1);  add_tensor = select_scatter_default = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(input_34, input_35, 3, 0, 257, 1);  input_34 = input_35 = None
        slice_scatter_default_1 = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_default, 2, 0, 256, 1);  full_default = slice_scatter_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, slice_scatter_default_1, alpha = 1);  add_tensor_1 = slice_scatter_default_1 = None
        view_default = torch.ops.aten.view.default(add_tensor_2, input_36);  add_tensor_2 = input_36 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default, [0, 0, 0, -1], 0);  view_default = None
        view_default_1 = torch.ops.aten.view.default(constant_pad_nd_default, input_37);  constant_pad_nd_default = input_37 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [0, 1, 2, 4, 3]);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(permute_default, input_38);  permute_default = input_38 = None
        bmm_default = torch.ops.aten.bmm.default(input_39, view_default_2);  input_39 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_2, input_40);  view_default_2 = input_40 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default, input_41);  bmm_default = input_41 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 1, 4, 3, 2]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default_1, input_42);  bmm_default_1 = input_42 = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 4, 2]);  permute_default_1 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(permute_default_2, 4);  permute_default_2 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(view_default_4, 4);  view_default_4 = None
        full_default_1 = torch.ops.aten.full.default(input_43, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_43 = None
        clone_default = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_44);  clone_default = None
        view_default_6 = torch.ops.aten.view.default(input_45, input_44);  input_45 = input_44 = None
        index_add_default = torch.ops.aten.index_add.default(full_default_1, 0, view_default_6, view_default_5, alpha = 1);  view_default_5 = None
        view_default_7 = torch.ops.aten.view.default(squeeze_dim_1, input_46);  squeeze_dim_1 = input_46 = None
        index_add_default_1 = torch.ops.aten.index_add.default(full_default_1, 0, view_default_6, view_default_7, alpha = 1);  full_default_1 = view_default_6 = view_default_7 = None
        as_strided_default = torch.ops.aten.as_strided.default(index_add_default_1, input_47, input_48, input_49);  index_add_default_1 = None
        view_default_8 = torch.ops.aten.view.default(as_strided_default, input_50);  as_strided_default = input_50 = None
        as_strided_default_1 = torch.ops.aten.as_strided.default(index_add_default, input_47, input_48, input_49);  index_add_default = input_47 = input_48 = input_49 = None
        return (view_default_8, as_strided_default_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
