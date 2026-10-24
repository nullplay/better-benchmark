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
    def forward(self, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67):
        squeeze_dim = torch.ops.aten.squeeze.dim(input_55, 4);  input_55 = None
        permute_default = torch.ops.aten.permute.default(input_56, [0, 1, 4, 3, 2]);  input_56 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [3, 0, 1, 4, 2]);  permute_default = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(permute_default_1, 4);  permute_default_1 = None
        add_tensor = torch.ops.aten.add.Tensor(input_57, input_58, alpha = 1);  input_57 = input_58 = None
        view_default = torch.ops.aten.view.default(squeeze_dim, input_59);  squeeze_dim = None
        permute_default_2 = torch.ops.aten.permute.default(view_default, [0, 1, 4, 2, 3]);  view_default = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_60);  clone_default = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        mm_default = torch.ops.aten.mm.default(input_61, squeeze_dim_2)
        unsqueeze_default = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        mm_default_1 = torch.ops.aten.mm.default(squeeze_dim_2, input_62);  squeeze_dim_2 = input_62 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(unsqueeze_default, input_63);  unsqueeze_default = None
        view_default_3 = torch.ops.aten.view.default(unsqueeze_default_1, input_64);  unsqueeze_default_1 = None
        squeeze_dim_3 = torch.ops.aten.squeeze.dim(view_default_2, 4);  view_default_2 = None
        squeeze_dim_4 = torch.ops.aten.squeeze.dim(squeeze_dim_3, 3);  squeeze_dim_3 = None
        squeeze_dim_5 = torch.ops.aten.squeeze.dim(view_default_3, 4);  view_default_3 = None
        squeeze_dim_6 = torch.ops.aten.squeeze.dim(squeeze_dim_5, 3);  squeeze_dim_5 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(squeeze_dim_4, torch.float32);  squeeze_dim_4 = None
        view_default_4 = torch.ops.aten.view.default(squeeze_dim_1, input_59);  squeeze_dim_1 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [0, 1, 4, 2, 3]);  view_default_4 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_60);  clone_default_1 = None
        squeeze_dim_7 = torch.ops.aten.squeeze.dim(view_default_5, 0);  view_default_5 = None
        mm_default_2 = torch.ops.aten.mm.default(input_61, squeeze_dim_7)
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(mm_default_2, 0);  mm_default_2 = None
        mm_default_3 = torch.ops.aten.mm.default(squeeze_dim_7, input_65);  squeeze_dim_7 = input_65 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(mm_default_3, 0);  mm_default_3 = None
        view_default_6 = torch.ops.aten.view.default(unsqueeze_default_2, input_63);  unsqueeze_default_2 = None
        view_default_7 = torch.ops.aten.view.default(unsqueeze_default_3, input_64);  unsqueeze_default_3 = None
        squeeze_dim_8 = torch.ops.aten.squeeze.dim(view_default_6, 4);  view_default_6 = None
        squeeze_dim_9 = torch.ops.aten.squeeze.dim(squeeze_dim_8, 3);  squeeze_dim_8 = None
        squeeze_dim_10 = torch.ops.aten.squeeze.dim(view_default_7, 4);  view_default_7 = None
        squeeze_dim_11 = torch.ops.aten.squeeze.dim(squeeze_dim_10, 3);  squeeze_dim_10 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(squeeze_dim_9, torch.float32);  squeeze_dim_9 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(squeeze_dim_11, torch.float32);  squeeze_dim_11 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_66, convert_element_type_default_2, alpha = 1);  input_66 = convert_element_type_default_2 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor, input_59);  add_tensor = input_59 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_8, [0, 1, 4, 2, 3]);  view_default_8 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_2, input_60);  clone_default_2 = input_60 = None
        squeeze_dim_12 = torch.ops.aten.squeeze.dim(view_default_9, 0);  view_default_9 = None
        mm_default_4 = torch.ops.aten.mm.default(input_61, squeeze_dim_12);  input_61 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(mm_default_4, 0);  mm_default_4 = None
        mm_default_5 = torch.ops.aten.mm.default(squeeze_dim_12, input_67);  squeeze_dim_12 = input_67 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(mm_default_5, 0);  mm_default_5 = None
        view_default_10 = torch.ops.aten.view.default(unsqueeze_default_4, input_63);  unsqueeze_default_4 = input_63 = None
        view_default_11 = torch.ops.aten.view.default(unsqueeze_default_5, input_64);  unsqueeze_default_5 = input_64 = None
        squeeze_dim_13 = torch.ops.aten.squeeze.dim(view_default_10, 4);  view_default_10 = None
        squeeze_dim_14 = torch.ops.aten.squeeze.dim(squeeze_dim_13, 3);  squeeze_dim_13 = None
        squeeze_dim_15 = torch.ops.aten.squeeze.dim(view_default_11, 4);  view_default_11 = None
        squeeze_dim_16 = torch.ops.aten.squeeze.dim(squeeze_dim_15, 3);  squeeze_dim_15 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(squeeze_dim_14, torch.float32);  squeeze_dim_14 = None
        return (squeeze_dim_6, convert_element_type_default, convert_element_type_default_1, add_tensor_1, squeeze_dim_16, convert_element_type_default_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
