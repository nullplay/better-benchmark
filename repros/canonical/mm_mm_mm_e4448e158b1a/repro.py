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
    def forward(self, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73):
        embedding_default = torch.ops.aten.embedding.default(input_54, input_55, -1, False, False);  input_54 = input_55 = None
        mm_default = torch.ops.aten.mm.default(input_56, input_57);  input_56 = input_57 = None
        view_default = torch.ops.aten.view.default(mm_default, input_58);  mm_default = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, view_default, alpha = 1);  embedding_default = view_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        rsqrt_default = torch.ops.aten.rsqrt.default(input_59);  input_59 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, rsqrt_default);  rsqrt_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        view_default_1 = torch.ops.aten.view.default(input_60, input_61);  input_60 = input_61 = None
        permute_default = torch.ops.aten.permute.default(input_62, [1, 0]);  input_62 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, permute_default);  view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(mm_default_1, input_58);  mm_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_2, alpha = 1);  add_tensor = view_default_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2);  convert_element_type_default_2 = None
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_1, rsqrt_default_1);  rsqrt_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_63, convert_element_type_default_3);  input_63 = convert_element_type_default_3 = None
        view_default_3 = torch.ops.aten.view.default(mul_tensor_2, input_64)
        permute_default_1 = torch.ops.aten.permute.default(input_65, [1, 0]);  input_65 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_3, permute_default_1);  view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_66);  mm_default_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_4, 0.5)
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(view_default_4, 3.0)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(pow_tensor_scalar_2, 0.044715);  pow_tensor_scalar_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_4, mul_tensor_4, alpha = 1);  view_default_4 = mul_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_3, 0.7978845608028654);  add_tensor_3 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_5);  mul_tensor_5 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_3, add_tensor_4);  mul_tensor_3 = add_tensor_4 = None
        view_default_5 = torch.ops.aten.view.default(mul_tensor_2, input_64);  mul_tensor_2 = input_64 = None
        permute_default_2 = torch.ops.aten.permute.default(input_67, [1, 0]);  input_67 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_5, permute_default_2);  view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_3, input_66);  mm_default_3 = input_66 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_6, view_default_6);  mul_tensor_6 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(mul_tensor_7, input_68);  mul_tensor_7 = input_68 = None
        permute_default_3 = torch.ops.aten.permute.default(input_69, [1, 0]);  input_69 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_7, permute_default_3);  view_default_7 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_4, input_58);  mm_default_4 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_8, alpha = 1);  add_tensor_1 = view_default_8 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        pow_tensor_scalar_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2);  convert_element_type_default_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(input_70);  input_70 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_tensor_5, rsqrt_default_2);  rsqrt_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.bfloat16);  mul_tensor_8 = None
        view_default_9 = torch.ops.aten.view.default(input_71, input_58);  input_71 = input_58 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_9, alpha = 1);  add_tensor_5 = view_default_9 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor_6, input_72);  input_72 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, input_73, alpha = 1);  add_tensor_6 = input_73 = None
        return (pow_tensor_scalar, convert_element_type_default_1, pow_tensor_scalar_3, convert_element_type_default_5, convert_element_type_default_6, mul_tensor_9, add_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
