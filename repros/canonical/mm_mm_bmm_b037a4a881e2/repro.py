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
    def forward(self, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69):
        embedding_default = torch.ops.aten.embedding.default(input_49, input_50, -1, False, False);  input_50 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(embedding_default, rsqrt_default);  rsqrt_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_51, convert_element_type_default_1);  input_51 = convert_element_type_default_1 = None
        view_default = torch.ops.aten.view.default(mul_tensor_1, input_52)
        permute_default = torch.ops.aten.permute.default(input_53, [1, 0]);  input_53 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_54);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_55);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_56, implicit = False);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(mul_tensor_1, input_52)
        permute_default_2 = torch.ops.aten.permute.default(input_57, [1, 0]);  input_57 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, permute_default_2);  view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_54);  mm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_55);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_3, [0, 1, 3, 2]);  permute_default_3 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_4, input_58, implicit = False);  permute_default_4 = input_58 = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_59);  clone_default = input_59 = None
        bmm_default = torch.ops.aten.bmm.default(input_60, view_default_6);  input_60 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(mul_tensor_1, input_52);  mul_tensor_1 = input_52 = None
        permute_default_5 = torch.ops.aten.permute.default(input_61, [1, 0]);  input_61 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_7, permute_default_5);  view_default_7 = permute_default_5 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_2, input_54);  mm_default_2 = input_54 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_55);  view_default_8 = input_55 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_9, [0, 2, 1, 3]);  view_default_9 = None
        expand_default_2 = torch.ops.aten.expand.default(permute_default_6, input_56, implicit = False);  permute_default_6 = input_56 = None
        mm_default_3 = torch.ops.aten.mm.default(input_62, input_63);  input_62 = input_63 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_3, input_64);  mm_default_3 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(embedding_default, view_default_10, alpha = 1);  embedding_default = view_default_10 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2);  convert_element_type_default_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(input_65);  input_65 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_1, rsqrt_default_1);  rsqrt_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        view_default_11 = torch.ops.aten.view.default(input_66, input_64);  input_66 = input_64 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_11, alpha = 1);  add_tensor_1 = view_default_11 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_2, input_67);  input_67 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, input_68, alpha = 1);  add_tensor_2 = input_68 = None
        permute_default_7 = torch.ops.aten.permute.default(input_49, [1, 0]);  input_49 = None
        mm_default_4 = torch.ops.aten.mm.default(input_69, permute_default_7);  input_69 = permute_default_7 = None
        return (expand_default, bmm_default, expand_default_2, pow_tensor_scalar_1, convert_element_type_default_3, convert_element_type_default_4, mul_tensor_3, add_tensor_3, mm_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
