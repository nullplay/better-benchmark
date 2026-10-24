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
    def forward(self, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90):
        add_tensor = torch.ops.aten.add.Tensor(input_63, input_64, alpha = 1);  input_63 = input_64 = None
        view_default = torch.ops.aten.view.default(input_65, input_66);  input_65 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        mm_default = torch.ops.aten.mm.default(input_67, input_68);  input_67 = input_68 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_66);  mm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2);  convert_element_type_default = None
        rsqrt_default = torch.ops.aten.rsqrt.default(input_69);  input_69 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_2, rsqrt_default);  rsqrt_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        view_default_2 = torch.ops.aten.view.default(input_70, input_71);  input_70 = input_71 = None
        permute_default = torch.ops.aten.permute.default(input_72, [1, 0]);  input_72 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, permute_default);  view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_66);  mm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2);  convert_element_type_default_2 = None
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar_1, [-1], True, dtype = None);  pow_tensor_scalar_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_3, rsqrt_default_1);  rsqrt_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_73, convert_element_type_default_3);  input_73 = convert_element_type_default_3 = None
        view_default_4 = torch.ops.aten.view.default(mul_tensor_2, input_74)
        permute_default_1 = torch.ops.aten.permute.default(input_75, [1, 0]);  input_75 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, permute_default_1);  view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_76);  mm_default_2 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_77);  view_default_5 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        expand_default = torch.ops.aten.expand.default(permute_default_2, input_78, implicit = False);  permute_default_2 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_79);  clone_default = input_79 = None
        view_default_8 = torch.ops.aten.view.default(mul_tensor_2, input_74)
        permute_default_3 = torch.ops.aten.permute.default(input_80, [1, 0]);  input_80 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_8, permute_default_3);  view_default_8 = permute_default_3 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_3, input_76);  mm_default_3 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_77);  view_default_9 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 1, 3, 2]);  permute_default_4 = None
        view_default_11 = torch.ops.aten.view.default(input_81, input_82);  input_81 = input_82 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_7, view_default_11);  view_default_7 = view_default_11 = None
        view_default_12 = torch.ops.aten.view.default(bmm_default, input_83);  bmm_default = input_83 = None
        view_default_13 = torch.ops.aten.view.default(mul_tensor_2, input_74);  mul_tensor_2 = input_74 = None
        permute_default_6 = torch.ops.aten.permute.default(input_84, [1, 0]);  input_84 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_13, permute_default_6);  view_default_13 = permute_default_6 = None
        view_default_14 = torch.ops.aten.view.default(mm_default_4, input_76);  mm_default_4 = input_76 = None
        view_default_15 = torch.ops.aten.view.default(view_default_14, input_77);  view_default_14 = input_77 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_15, [0, 2, 1, 3]);  view_default_15 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_7, input_78, implicit = False);  permute_default_7 = input_78 = None
        mm_default_5 = torch.ops.aten.mm.default(input_85, input_86);  input_85 = input_86 = None
        view_default_16 = torch.ops.aten.view.default(mm_default_5, input_66);  mm_default_5 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_3, view_default_16, alpha = 1);  add_tensor_3 = view_default_16 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_4, 2);  convert_element_type_default_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(input_87);  input_87 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_5, rsqrt_default_2);  rsqrt_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default_17 = torch.ops.aten.view.default(input_88, input_66);  input_88 = input_66 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_17, alpha = 1);  add_tensor_5 = view_default_17 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_6, input_89);  input_89 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, input_90, alpha = 1);  add_tensor_6 = input_90 = None
        return (pow_tensor_scalar, convert_element_type_default_1, permute_default_5, view_default_12, expand_default_1, pow_tensor_scalar_2, convert_element_type_default_5, convert_element_type_default_6, mul_tensor_4, add_tensor_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
