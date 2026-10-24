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
    def forward(self, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62):
        embedding_default = torch.ops.aten.embedding.default(input_51, input_52, -1, False, False);  input_51 = input_52 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_53, 0.1);  input_53 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, embedding_default);  gt_scalar = embedding_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_54, torch.bfloat16);  input_54 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(convert_element_type_default, 3);  convert_element_type_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 3);  convert_element_type_default_1 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        view_default = torch.ops.aten.view.default(unsqueeze_default_1, input_55);  unsqueeze_default_1 = input_55 = None
        view_default_1 = torch.ops.aten.view.default(unsqueeze_default_3, input_56);  unsqueeze_default_3 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default, 0);  view_default = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        mm_default = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_1);  squeeze_dim_1 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(unsqueeze_default_4, input_57);  unsqueeze_default_4 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 1, 3, 4, 2]);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(permute_default, input_58);  permute_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_59, torch.bfloat16);  input_59 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(convert_element_type_default_2, 3);  convert_element_type_default_2 = None
        unsqueeze_default_6 = torch.ops.aten.unsqueeze.default(unsqueeze_default_5, 4);  unsqueeze_default_5 = None
        view_default_4 = torch.ops.aten.view.default(unsqueeze_default_6, input_56);  unsqueeze_default_6 = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(view_default_4, 0);  view_default_4 = None
        mm_default_1 = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_2);  squeeze_dim_2 = None
        unsqueeze_default_7 = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(unsqueeze_default_7, input_57);  unsqueeze_default_7 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_5, [0, 1, 3, 4, 2]);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(permute_default_1, input_58);  permute_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_60, torch.bfloat16);  input_60 = None
        unsqueeze_default_8 = torch.ops.aten.unsqueeze.default(convert_element_type_default_3, 3);  convert_element_type_default_3 = None
        unsqueeze_default_9 = torch.ops.aten.unsqueeze.default(unsqueeze_default_8, 4);  unsqueeze_default_8 = None
        view_default_7 = torch.ops.aten.view.default(unsqueeze_default_9, input_56);  unsqueeze_default_9 = input_56 = None
        squeeze_dim_3 = torch.ops.aten.squeeze.dim(view_default_7, 0);  view_default_7 = None
        mm_default_2 = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_3);  squeeze_dim = squeeze_dim_3 = None
        unsqueeze_default_10 = torch.ops.aten.unsqueeze.default(mm_default_2, 0);  mm_default_2 = None
        view_default_8 = torch.ops.aten.view.default(unsqueeze_default_10, input_57);  unsqueeze_default_10 = input_57 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_8, [0, 1, 3, 4, 2]);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(permute_default_2, input_58);  permute_default_2 = input_58 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_3, input_61, alpha = 1);  input_61 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        unsqueeze_default_11 = torch.ops.aten.unsqueeze.default(convert_element_type_default_4, 4);  convert_element_type_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(unsqueeze_default_11, [1, 2, 0, 4, 3]);  unsqueeze_default_11 = None
        unsqueeze_default_12 = torch.ops.aten.unsqueeze.default(view_default_6, 4);  view_default_6 = None
        permute_default_4 = torch.ops.aten.permute.default(unsqueeze_default_12, [1, 2, 4, 0, 3]);  unsqueeze_default_12 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_3, input_62, alpha = 1);  view_default_3 = input_62 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        unsqueeze_default_13 = torch.ops.aten.unsqueeze.default(convert_element_type_default_5, 4);  convert_element_type_default_5 = None
        permute_default_5 = torch.ops.aten.permute.default(unsqueeze_default_13, [1, 2, 0, 4, 3]);  unsqueeze_default_13 = None
        unsqueeze_default_14 = torch.ops.aten.unsqueeze.default(view_default_9, 4);  view_default_9 = None
        permute_default_6 = torch.ops.aten.permute.default(unsqueeze_default_14, [4, 1, 2, 3, 0]);  unsqueeze_default_14 = None
        return (permute_default_3, permute_default_4, permute_default_5, permute_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
