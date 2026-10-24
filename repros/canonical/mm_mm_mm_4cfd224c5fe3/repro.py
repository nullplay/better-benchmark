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
    def forward(self, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68):
        embedding_default = torch.ops.aten.embedding.default(input_50, input_51, -1, False, False);  input_50 = input_51 = None
        view_default = torch.ops.aten.view.default(input_52, input_53);  input_52 = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, view_default, alpha = 1);  embedding_default = view_default = None
        mm_default = torch.ops.aten.mm.default(input_54, input_55);  input_54 = input_55 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_53);  mm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_1, alpha = 1);  add_tensor = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_56, convert_element_type_default_1);  input_56 = convert_element_type_default_1 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_1, input_57)
        permute_default = torch.ops.aten.permute.default(input_58, [1, 0]);  input_58 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, permute_default);  view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_59);  mm_default_1 = input_59 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_60);  view_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06, alpha = 1);  mean_dim_1 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default_1);  convert_element_type_default_2 = rsqrt_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_61, convert_element_type_default_3);  input_61 = convert_element_type_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(mul_tensor_3, [0, 2, 1, 3]);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(permute_default_1, input_62);  input_62 = None
        slice_tensor = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 9223372036854775807, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64, 1);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mul_tensor_1, input_57)
        permute_default_2 = torch.ops.aten.permute.default(input_63, [1, 0]);  input_63 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_5, permute_default_2);  view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_64);  mm_default_2 = input_64 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_60);  view_default_6 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_65);  convert_element_type_default_4 = input_65 = None
        view_default_8 = torch.ops.aten.view.default(mul_tensor_1, input_57);  mul_tensor_1 = input_57 = None
        permute_default_3 = torch.ops.aten.permute.default(input_66, [1, 0]);  input_66 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_8, permute_default_3);  view_default_8 = permute_default_3 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_3, input_53);  mm_default_3 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_60);  view_default_9 = input_60 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(permute_default_4, 2)
        view_default_11 = torch.ops.aten.view.default(input_67, input_53);  input_67 = input_53 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_11, alpha = 1);  add_tensor_1 = view_default_11 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, input_68, alpha = 1);  add_tensor_4 = input_68 = None
        return (mul_tensor_4, slice_tensor, slice_tensor_1, pow_tensor_scalar_2, mul_tensor_5, permute_default_4, unsqueeze_default, convert_element_type_default_5, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
