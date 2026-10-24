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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59):
        mul_tensor = torch.ops.aten.mul.Tensor(input_44, input_45);  input_44 = input_45 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_46, torch.bfloat16);  input_46 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default, alpha = 1);  mul_tensor = convert_element_type_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_47, input_48);  input_47 = input_48 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_1, alpha = 1);  add_tensor = convert_element_type_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default);  convert_element_type_default_2 = rsqrt_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_49, torch.float32);  input_49 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(convert_element_type_default_3, 1.0, alpha = 1);  convert_element_type_default_3 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_3);  mul_tensor_2 = add_tensor_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_4, input_50)
        permute_default = torch.ops.aten.permute.default(input_51, [1, 0]);  input_51 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_52);  mm_default = input_52 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_53);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(permute_default_1, input_54)
        slice_tensor = torch.ops.aten.slice.Tensor(permute_default_1, 3, 128, 9223372036854775807, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 128, 1);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_4, input_50)
        permute_default_2 = torch.ops.aten.permute.default(input_55, [1, 0]);  input_55 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, permute_default_2);  view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_56);  mm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_53);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(permute_default_3, input_54);  input_54 = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(permute_default_3, 3, 128, 9223372036854775807, 1)
        slice_tensor_3 = torch.ops.aten.slice.Tensor(permute_default_3, 3, 0, 128, 1);  permute_default_3 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_4, input_50);  convert_element_type_default_4 = input_50 = None
        permute_default_4 = torch.ops.aten.permute.default(input_57, [1, 0]);  input_57 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_6, permute_default_4);  view_default_6 = permute_default_4 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_2, input_56);  mm_default_2 = input_56 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_53);  view_default_7 = input_53 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(permute_default_5, 2)
        expand_default = torch.ops.aten.expand.default(unsqueeze_default, input_58, implicit = False);  unsqueeze_default = input_58 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_9 = torch.ops.aten.view.default(clone_default, input_59);  clone_default = input_59 = None
        return (mul_tensor_4, slice_tensor, slice_tensor_1, mul_tensor_5, slice_tensor_2, slice_tensor_3, permute_default_5, view_default_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
