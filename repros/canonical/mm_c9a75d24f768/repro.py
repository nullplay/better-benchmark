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
    def forward(self, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39):
        add_tensor = torch.ops.aten.add.Tensor(input_29, input_30, alpha = 1);  input_29 = input_30 = None
        permute_default = torch.ops.aten.permute.default(input_31, [2, 0, 1, 4, 3]);  input_31 = None
        view_default = torch.ops.aten.view.default(permute_default, input_32);  permute_default = input_32 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_33, torch.bfloat16);  input_33 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(view_default, 4);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(unsqueeze_default, [0, 1, 4, 3, 2]);  unsqueeze_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(convert_element_type_default, 3);  convert_element_type_default = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 4);  unsqueeze_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(unsqueeze_default_2, [3, 4, 0, 2, 1]);  unsqueeze_default_2 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_1, [0, 1, 3, 4, 2]);  permute_default_1 = None
        clone_default = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_34);  clone_default = input_34 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_2, [3, 4, 2, 0, 1]);  permute_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        view_default_2 = torch.ops.aten.view.default(clone_default_1, input_35);  clone_default_1 = input_35 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(view_default_2, 0);  view_default_2 = None
        mm_default = torch.ops.aten.mm.default(squeeze_dim, squeeze_dim_1);  squeeze_dim = squeeze_dim_1 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        view_default_3 = torch.ops.aten.view.default(unsqueeze_default_3, input_36);  unsqueeze_default_3 = input_36 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_3, [0, 1, 4, 2, 3]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(permute_default_5, input_37);  permute_default_5 = input_37 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_38, torch.bfloat16);  input_38 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_1, 0.1);  convert_element_type_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, view_default_4);  gt_scalar = view_default_4 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, add_tensor, alpha = 1);  mul_tensor_1 = add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, input_39, alpha = 1);  add_tensor_1 = input_39 = None
        return (var_mean_correction, sub_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
