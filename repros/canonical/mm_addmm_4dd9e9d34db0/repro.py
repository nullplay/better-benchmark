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
    def forward(self, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36):
        permute_default = torch.ops.aten.permute.default(input_26, [1, 0]);  input_26 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        embedding_default = torch.ops.aten.embedding.default(input_27, clone_default, -1, False, False);  clone_default = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(embedding_default, 3);  embedding_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        view_default = torch.ops.aten.view.default(unsqueeze_default_1, input_28);  unsqueeze_default_1 = input_28 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default, 0);  view_default = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(input_29, 3);  input_29 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        view_default_1 = torch.ops.aten.view.default(unsqueeze_default_3, input_30);  unsqueeze_default_3 = input_30 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        mm_default = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_1);  squeeze_dim = squeeze_dim_1 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(unsqueeze_default_4, input_31);  unsqueeze_default_4 = input_31 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 4, 2]);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(permute_default_1, input_32);  permute_default_1 = input_32 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_3, input_33, alpha = 1);  input_33 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(add_tensor, 4);  add_tensor = None
        permute_default_2 = torch.ops.aten.permute.default(unsqueeze_default_5, [1, 2, 0, 4, 3]);  unsqueeze_default_5 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 2, 4, 3]);  permute_default_2 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_3, input_34, alpha = 1);  view_default_3 = input_34 = None
        unsqueeze_default_6 = torch.ops.aten.unsqueeze.default(add_tensor_1, 4);  add_tensor_1 = None
        permute_default_4 = torch.ops.aten.permute.default(unsqueeze_default_6, [1, 2, 0, 4, 3]);  unsqueeze_default_6 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 1, 2, 4, 3]);  permute_default_4 = None
        permute_default_6 = torch.ops.aten.permute.default(input_27, [1, 0]);  input_27 = None
        addmm_default = torch.ops.aten.addmm.default(input_35, input_36, permute_default_6, beta = 1, alpha = 1);  input_35 = input_36 = permute_default_6 = None
        return (permute_default_3, permute_default_5, addmm_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
