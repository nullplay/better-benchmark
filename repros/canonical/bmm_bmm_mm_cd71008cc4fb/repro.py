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
    def forward(self, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80):
        add_tensor = torch.ops.aten.add.Tensor(input_55, input_56, alpha = 1);  input_55 = input_56 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_57, input_58);  input_57 = input_58 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        view_default = torch.ops.aten.view.default(input_59, input_60);  input_59 = input_60 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_61);  view_default = input_61 = None
        bmm_default = torch.ops.aten.bmm.default(input_62, view_default_1);  input_62 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_1, input_63);  view_default_1 = input_63 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default_1, input_64);  bmm_default_1 = input_64 = None
        view_default_3 = torch.ops.aten.view.default(input_65, input_66);  input_65 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_67);  view_default_3 = None
        permute_default = torch.ops.aten.permute.default(view_default_4, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_68);  permute_default = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_4, input_69);  view_default_4 = input_69 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_70);  mm_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        clone_default = torch.ops.aten.clone.default(input_71, memory_format = torch.contiguous_format);  input_71 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_67);  clone_default = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_6, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_68);  permute_default_1 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_6, input_72);  view_default_6 = input_72 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_3, input_70);  mm_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default, convert_element_type_default_2, alpha = 1);  convert_element_type_default = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_1, input_66);  clone_default_1 = input_66 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_67);  view_default_8 = input_67 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_9, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_3, input_68);  permute_default_3 = input_68 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_9, input_73);  view_default_9 = input_73 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_5, input_70);  mm_default_5 = input_70 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_4, alpha = 1);  add_tensor_2 = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_3, input_74);  input_74 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_75, input_76);  input_75 = input_76 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, input_77)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_3, mul_tensor_4);  add_tensor_3 = mul_tensor_4 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], True, dtype = torch.float32);  mul_tensor_5 = None
        view_default_11 = torch.ops.aten.view.default(sum_dim_int_list, input_78);  sum_dim_int_list = input_78 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_1, mul_tensor_3)
        mul_tensor_7 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_77);  mul_tensor_1 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [2], True, dtype = torch.float32);  mul_tensor_6 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_7, alpha = 1);  add_tensor_1 = mul_tensor_7 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_77, 3);  input_77 = None
        mul_scalar = torch.ops.aten.mul.Scalar(sum_dim_int_list_1, -0.5);  sum_dim_int_list_1 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_scalar, pow_tensor_scalar);  mul_scalar = pow_tensor_scalar = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(mul_tensor_3, 1.0);  mul_tensor_3 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(input_79, mul_scalar_1);  input_79 = mul_scalar_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_9, alpha = 1);  add_tensor_4 = mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(add_tensor_5, input_80);  add_tensor_5 = input_80 = None
        return (bmm_default, convert_element_type_default_1, convert_element_type_default_3, convert_element_type_default_5, view_default_11, mul_tensor_8, mul_tensor_10)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
