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
    def forward(self, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102, input_103, input_104, input_105, input_106, input_107, input_108, input_109, input_110, input_111, input_112, input_113, input_114, input_115, input_116, input_117, input_118, input_119):
        add_tensor = torch.ops.aten.add.Tensor(input_90, input_91, alpha = 1);  input_90 = input_91 = None
        view_default = torch.ops.aten.view.default(input_92, input_93);  input_92 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        mm_default = torch.ops.aten.mm.default(input_94, input_95);  input_94 = input_95 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_93);  mm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_96);  convert_element_type_default = input_96 = None
        view_default_2 = torch.ops.aten.view.default(input_97, input_98);  input_97 = None
        permute_default = torch.ops.aten.permute.default(input_99, [1, 0]);  input_99 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, permute_default);  view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_93);  mm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_1, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(input_100);  input_100 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, rsqrt_default);  convert_element_type_default_1 = rsqrt_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default_4 = torch.ops.aten.view.default(input_101, input_102);  input_101 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_103);  view_default_4 = None
        permute_default_1 = torch.ops.aten.permute.default(input_104, [1, 0]);  input_104 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_5, permute_default_1);  view_default_5 = permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(mm_default_2, input_93);  mm_default_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_6, alpha = 1);  add_tensor_3 = view_default_6 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.float32)
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_3, 2)
        mean_dim_1 = torch.ops.aten.mean.dim(pow_tensor_scalar_2, [-1], True, dtype = None);  pow_tensor_scalar_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mean_dim_1, 1e-06, alpha = 1);  mean_dim_1 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, rsqrt_default_1);  convert_element_type_default_3 = rsqrt_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_105, convert_element_type_default_4);  input_105 = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_106, torch.bfloat16);  input_106 = None
        view_default_7 = torch.ops.aten.view.default(input_107, input_108);  input_107 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, view_default_7);  convert_element_type_default_5 = view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(mul_tensor_4, input_98);  mul_tensor_4 = None
        permute_default_2 = torch.ops.aten.permute.default(input_109, [1, 0]);  input_109 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_8, permute_default_2);  view_default_8 = permute_default_2 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_3, input_93);  mm_default_3 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_4, view_default_9, alpha = 1);  add_tensor_4 = view_default_9 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        pow_tensor_scalar_3 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_6, 2)
        mean_dim_2 = torch.ops.aten.mean.dim(pow_tensor_scalar_3, [-1], True, dtype = None);  pow_tensor_scalar_3 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mean_dim_2, 1e-06, alpha = 1);  mean_dim_2 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_7);  add_tensor_7 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_6, rsqrt_default_2);  convert_element_type_default_6 = rsqrt_default_2 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_110, convert_element_type_default_7);  input_110 = convert_element_type_default_7 = None
        view_default_10 = torch.ops.aten.view.default(mul_tensor_6, input_111)
        view_default_11 = torch.ops.aten.view.default(mul_tensor_6, input_111)
        view_default_12 = torch.ops.aten.view.default(mul_tensor_6, input_111);  mul_tensor_6 = None
        getitem = input_112[0];  input_112 = None
        permute_default_3 = torch.ops.aten.permute.default(getitem, [0, 2, 1, 3]);  getitem = None
        view_default_13 = torch.ops.aten.view.default(permute_default_3, input_102);  permute_default_3 = input_102 = None
        view_default_14 = torch.ops.aten.view.default(view_default_13, input_103);  view_default_13 = input_103 = None
        permute_default_4 = torch.ops.aten.permute.default(input_113, [1, 0]);  input_113 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_14, permute_default_4);  view_default_14 = permute_default_4 = None
        view_default_15 = torch.ops.aten.view.default(mm_default_4, input_93);  mm_default_4 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_6, view_default_15, alpha = 1);  add_tensor_6 = view_default_15 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.float32)
        pow_tensor_scalar_4 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_8, 2)
        mean_dim_3 = torch.ops.aten.mean.dim(pow_tensor_scalar_4, [-1], True, dtype = None);  pow_tensor_scalar_4 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(mean_dim_3, 1e-06, alpha = 1);  mean_dim_3 = None
        rsqrt_default_3 = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, rsqrt_default_3);  convert_element_type_default_8 = rsqrt_default_3 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(input_114, convert_element_type_default_9);  input_114 = convert_element_type_default_9 = None
        view_default_16 = torch.ops.aten.view.default(mul_tensor_8, input_111)
        mm_default_5 = torch.ops.aten.mm.default(view_default_16, input_115);  view_default_16 = input_115 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_116, torch.bfloat16);  input_116 = None
        view_default_17 = torch.ops.aten.view.default(mul_tensor_8, input_111);  mul_tensor_8 = None
        mm_default_6 = torch.ops.aten.mm.default(view_default_17, input_117);  view_default_17 = input_117 = None
        view_default_18 = torch.ops.aten.view.default(mm_default_6, input_108);  mm_default_6 = input_108 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(convert_element_type_default_10, view_default_18);  convert_element_type_default_10 = view_default_18 = None
        view_default_19 = torch.ops.aten.view.default(mul_tensor_9, input_98);  mul_tensor_9 = input_98 = None
        permute_default_5 = torch.ops.aten.permute.default(input_118, [1, 0]);  input_118 = None
        mm_default_7 = torch.ops.aten.mm.default(view_default_19, permute_default_5);  view_default_19 = permute_default_5 = None
        view_default_20 = torch.ops.aten.view.default(mm_default_7, input_93);  mm_default_7 = input_93 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(add_tensor_8, view_default_20, alpha = 1);  add_tensor_8 = view_default_20 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.float32);  add_tensor_10 = None
        pow_tensor_scalar_5 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_11, 2)
        mean_dim_4 = torch.ops.aten.mean.dim(pow_tensor_scalar_5, [-1], True, dtype = None);  pow_tensor_scalar_5 = None
        add_tensor_11 = torch.ops.aten.add.Tensor(mean_dim_4, 1e-06, alpha = 1);  mean_dim_4 = None
        rsqrt_default_4 = torch.ops.aten.rsqrt.default(add_tensor_11);  add_tensor_11 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_11, rsqrt_default_4);  convert_element_type_default_11 = rsqrt_default_4 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(mul_tensor_10, torch.bfloat16);  mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_119, convert_element_type_default_12);  input_119 = convert_element_type_default_12 = None
        view_default_21 = torch.ops.aten.view.default(mul_tensor_11, input_111);  mul_tensor_11 = input_111 = None
        return (pow_tensor_scalar, mul_tensor, mean_dim, convert_element_type_default_2, mul_tensor_3, view_default_10, view_default_11, view_default_12, mm_default_5, view_default_21)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
