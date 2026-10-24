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
    def forward(self, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74):
        embedding_default = torch.ops.aten.embedding.default(input_56, input_57, -1, False, False);  input_56 = input_57 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_59, torch.bfloat16);  input_59 = None
        view_default = torch.ops.aten.view.default(input_60, input_61);  input_60 = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, view_default, alpha = 1);  embedding_default = view_default = None
        mm_default = torch.ops.aten.mm.default(input_62, input_63);  input_62 = input_63 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_61);  mm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_1, alpha = 1);  add_tensor = view_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mean_dim, 1e-05, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_2, rsqrt_default);  convert_element_type_default_2 = rsqrt_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_64, convert_element_type_default_3);  input_64 = convert_element_type_default_3 = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor_1, input_65)
        permute_default = torch.ops.aten.permute.default(input_66, [1, 0]);  input_66 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, permute_default);  view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_61);  mm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_67);  view_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(convert_element_type_default, 1);  convert_element_type_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(permute_default_1, unsqueeze_default)
        slice_tensor = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 9223372036854775807, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64, 1);  permute_default_1 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_68, unsqueeze_default_1);  input_68 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_2, mul_tensor_3, alpha = 1);  mul_tensor_2 = mul_tensor_3 = None
        view_default_5 = torch.ops.aten.view.default(mul_tensor_1, input_65)
        permute_default_2 = torch.ops.aten.permute.default(input_69, [1, 0]);  input_69 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_5, permute_default_2);  view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_70);  mm_default_2 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_67);  view_default_6 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(permute_default_3, unsqueeze_default);  unsqueeze_default = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(permute_default_3, 3, 64, 9223372036854775807, 1)
        neg_default = torch.ops.aten.neg.default(slice_tensor_2);  slice_tensor_2 = None
        slice_tensor_3 = torch.ops.aten.slice.Tensor(permute_default_3, 3, 0, 64, 1);  permute_default_3 = None
        cat_default = torch.ops.aten.cat.default([neg_default, slice_tensor_3], -1);  neg_default = slice_tensor_3 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(cat_default, unsqueeze_default_1);  cat_default = unsqueeze_default_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_4, mul_tensor_5, alpha = 1);  mul_tensor_4 = mul_tensor_5 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(add_tensor_4, 2)
        expand_default = torch.ops.aten.expand.default(unsqueeze_default_2, input_71, implicit = False);  unsqueeze_default_2 = input_71 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_8 = torch.ops.aten.view.default(mul_tensor_1, input_65);  mul_tensor_1 = input_65 = None
        permute_default_4 = torch.ops.aten.permute.default(input_72, [1, 0]);  input_72 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_8, permute_default_4);  view_default_8 = permute_default_4 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_3, input_70);  mm_default_3 = input_70 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_67);  view_default_9 = input_67 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(permute_default_5, 2)
        view_default_11 = torch.ops.aten.view.default(input_73, input_61);  input_73 = input_61 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_11, alpha = 1);  add_tensor_1 = view_default_11 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, input_74, alpha = 1);  add_tensor_5 = input_74 = None
        return (slice_tensor, slice_tensor_1, add_tensor_3, add_tensor_4, clone_default, permute_default_5, unsqueeze_default_3, convert_element_type_default_4, add_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
