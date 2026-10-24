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
        sub_tensor = torch.ops.aten.sub.Tensor(input_26, input_27, alpha = 1);  input_26 = input_27 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_28, sub_tensor);  input_28 = sub_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_29, torch.bfloat16);  input_29 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_1);  convert_element_type_default = mul_tensor_1 = None
        view_default = torch.ops.aten.view.default(mul_tensor_2, input_30);  mul_tensor_2 = input_30 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 3, 4, 2]);  view_default = None
        view_default_1 = torch.ops.aten.view.default(permute_default, input_31);  permute_default = input_31 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        mm_default = torch.ops.aten.mm.default(input_32, squeeze_dim);  input_32 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        mm_default_1 = torch.ops.aten.mm.default(squeeze_dim, input_33);  squeeze_dim = input_33 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(unsqueeze_default, input_34);  unsqueeze_default = input_34 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [3, 4, 2, 0, 1]);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(unsqueeze_default_1, input_35);  unsqueeze_default_1 = input_35 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_3, [0, 1, 4, 2, 3]);  view_default_3 = None
        permute_default_3 = torch.ops.aten.permute.default(permute_default_1, [2, 4, 3, 0, 1]);  permute_default_1 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(permute_default_3, 4);  permute_default_3 = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(squeeze_dim_1, 3);  squeeze_dim_1 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_2, [0, 1, 4, 3, 2]);  permute_default_2 = None
        squeeze_dim_3 = torch.ops.aten.squeeze.dim(permute_default_4, 4);  permute_default_4 = None
        view_default_4 = torch.ops.aten.view.default(squeeze_dim_3, input_36);  squeeze_dim_3 = input_36 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_4, [1, 2, 0, 4, 3]);  view_default_4 = None
        clone_default = torch.ops.aten.clone.default(permute_default_5, memory_format = torch.contiguous_format);  permute_default_5 = None
        return (squeeze_dim_2, clone_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
