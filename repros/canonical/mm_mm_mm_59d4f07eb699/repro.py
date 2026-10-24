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
    def forward(self, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114):
        add_tensor = torch.ops.aten.add.Tensor(input_88, input_89, alpha = 1);  input_88 = input_89 = None
        view_default = torch.ops.aten.view.default(input_90, input_91);  input_90 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        mm_default = torch.ops.aten.mm.default(input_92, input_93);  input_92 = input_93 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_91);  mm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        rsqrt_default = torch.ops.aten.rsqrt.default(input_94);  input_94 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_2, rsqrt_default);  rsqrt_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        view_default_2 = torch.ops.aten.view.default(input_95, input_96);  input_95 = None
        permute_default = torch.ops.aten.permute.default(input_97, [1, 0]);  input_97 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, permute_default);  view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_91);  mm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2);  convert_element_type_default_2 = None
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_3, rsqrt_default_1);  rsqrt_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_98, convert_element_type_default_3);  input_98 = convert_element_type_default_3 = None
        view_default_4 = torch.ops.aten.view.default(input_99, input_100);  input_99 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_101);  view_default_4 = None
        permute_default_1 = torch.ops.aten.permute.default(input_102, [1, 0]);  input_102 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_5, permute_default_1);  view_default_5 = permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_91);  mm_default_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_6, alpha = 1);  add_tensor_3 = view_default_6 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2);  convert_element_type_default_4 = None
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_2, [-1], True, dtype = None);  pow_tensor_scalar_2 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06, alpha = 1);  mean_dim_1 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_5, rsqrt_default_2);  rsqrt_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_103, convert_element_type_default_5);  input_103 = convert_element_type_default_5 = None
        view_default_7 = torch.ops.aten.view.default(mul_tensor_4, input_104);  mul_tensor_4 = None
        clone_default = torch.ops.aten.clone.default(input_105, memory_format = torch.contiguous_format);  input_105 = None
        view_default_8 = torch.ops.aten.view.default(clone_default, input_100);  clone_default = input_100 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_101);  view_default_8 = input_101 = None
        permute_default_2 = torch.ops.aten.permute.default(input_106, [1, 0]);  input_106 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_9, permute_default_2);  view_default_9 = permute_default_2 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_3, input_91);  mm_default_3 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_10, alpha = 1);  add_tensor_5 = view_default_10 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.float32)
        pow_tensor_scalar_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_6, 2);  convert_element_type_default_6 = None
        mean_dim_2 = torch.ops.aten.mean.dim(pow_tensor_scalar_3, [-1], True, dtype = None);  pow_tensor_scalar_3 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mean_dim_2, 1e-06, alpha = 1);  mean_dim_2 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_8);  add_tensor_8 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_7, rsqrt_default_3);  rsqrt_default_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_107, convert_element_type_default_7);  input_107 = convert_element_type_default_7 = None
        view_default_11 = torch.ops.aten.view.default(mul_tensor_6, input_104)
        permute_default_3 = torch.ops.aten.permute.default(input_108, [1, 0]);  input_108 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_11, permute_default_3);  view_default_11 = permute_default_3 = None
        view_default_12 = torch.ops.aten.view.default(mm_default_4, input_109);  mm_default_4 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_12, 0.5)
        pow_tensor_scalar_4 = torch.ops.aten.pow.Tensor_Scalar(view_default_12, 3.0)
        mul_tensor_8 = torch.ops.aten.mul.Tensor(pow_tensor_scalar_4, 0.044715);  pow_tensor_scalar_4 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(view_default_12, mul_tensor_8, alpha = 1);  view_default_12 = mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor_9, 0.7978845608028654);  add_tensor_9 = None
        tanh_default = torch.ops.aten.tanh.default(mul_tensor_9);  mul_tensor_9 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_7, add_tensor_10);  mul_tensor_7 = add_tensor_10 = None
        view_default_13 = torch.ops.aten.view.default(mul_tensor_6, input_104);  mul_tensor_6 = None
        permute_default_4 = torch.ops.aten.permute.default(input_110, [1, 0]);  input_110 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_13, permute_default_4);  view_default_13 = permute_default_4 = None
        view_default_14 = torch.ops.aten.view.default(mm_default_5, input_109);  mm_default_5 = input_109 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(mul_tensor_10, view_default_14);  mul_tensor_10 = view_default_14 = None
        view_default_15 = torch.ops.aten.view.default(mul_tensor_11, input_96);  mul_tensor_11 = input_96 = None
        permute_default_5 = torch.ops.aten.permute.default(input_111, [1, 0]);  input_111 = None
        mm_default_6 = torch.ops.aten.mm.default(view_default_15, permute_default_5);  view_default_15 = permute_default_5 = None
        view_default_16 = torch.ops.aten.view.default(mm_default_6, input_91);  mm_default_6 = input_91 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(add_tensor_7, view_default_16, alpha = 1);  add_tensor_7 = view_default_16 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_11, torch.float32)
        pow_tensor_scalar_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_8, 2);  convert_element_type_default_8 = None
        mean_dim_3 = torch.ops.aten.mean.dim(pow_tensor_scalar_5, [-1], True, dtype = None);  pow_tensor_scalar_5 = None
        add_tensor_12 = torch.ops.aten.add.Tensor(mean_dim_3, 1e-06, alpha = 1);  mean_dim_3 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_12);  add_tensor_12 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(add_tensor_11, rsqrt_default_4);  add_tensor_11 = rsqrt_default_4 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.bfloat16);  mul_tensor_12 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(input_112, convert_element_type_default_9);  input_112 = convert_element_type_default_9 = None
        view_default_17 = torch.ops.aten.view.default(mul_tensor_13, input_104);  mul_tensor_13 = input_104 = None
        permute_default_6 = torch.ops.aten.permute.default(input_113, [1, 0]);  input_113 = None
        mm_default_7 = torch.ops.aten.mm.default(view_default_17, permute_default_6);  view_default_17 = permute_default_6 = None
        view_default_18 = torch.ops.aten.view.default(mm_default_7, input_114);  mm_default_7 = input_114 = None
        return (pow_tensor_scalar, convert_element_type_default_1, mul_tensor_2, view_default_7, view_default_18)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
