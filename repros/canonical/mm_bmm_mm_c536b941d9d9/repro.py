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
    def forward(self, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83):
        mul_tensor = torch.ops.aten.mul.Tensor(input_62, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_62, [2], True, dtype = None);  input_62 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(input_63, [2], True, dtype = None);  input_63 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_64, sum_dim_int_list_1);  input_64 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor, sum_dim_int_list, alpha = 1);  mul_tensor = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_1, alpha = 1);  sub_tensor = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_65, sub_tensor_1);  input_65 = sub_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16)
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_66);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_67);  view_default = input_67 = None
        bmm_default = torch.ops.aten.bmm.default(input_68, input_69);  input_68 = input_69 = None
        view_default_1 = torch.ops.aten.view.default(bmm_default, input_70);  bmm_default = input_70 = None
        mul_scalar = torch.ops.aten.mul.Scalar(view_default_1, 0.3535533905932738);  view_default_1 = None
        permute_default = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        view_default_2 = torch.ops.aten.view.default(input_71, input_72);  input_71 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_66);  view_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_73);  input_73 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [1, 0]);  view_default_3 = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_74);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_72);  mm_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_2, convert_element_type_default_1, alpha = 1);  mul_tensor_2 = convert_element_type_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        permute_default_2 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_5 = torch.ops.aten.view.default(permute_default_2, input_72);  permute_default_2 = None
        clone_default = torch.ops.aten.clone.default(view_default_5, memory_format = torch.contiguous_format);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_66);  clone_default = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_6, input_75);  input_75 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_6, [1, 0])
        mm_default_4 = torch.ops.aten.mm.default(permute_default_3, input_74);  permute_default_3 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_6, [0], True, dtype = torch.float32);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(sum_dim_int_list_2, input_76);  sum_dim_int_list_2 = input_76 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_7, torch.bfloat16);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_3, input_72);  mm_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_4, alpha = 1);  add_tensor = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float32);  convert_element_type_default_3 = None
        view_default_9 = torch.ops.aten.view.default(input_77, input_72);  input_77 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_66);  view_default_9 = input_66 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_10, input_78);  input_78 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_10, [1, 0]);  view_default_10 = None
        mm_default_6 = torch.ops.aten.mm.default(permute_default_4, input_74);  permute_default_4 = input_74 = None
        view_default_11 = torch.ops.aten.view.default(mm_default_5, input_72);  mm_default_5 = input_72 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32);  view_default_11 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_7, alpha = 1);  add_tensor_1 = convert_element_type_default_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_79, torch.float32);  input_79 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_9, 1.1111111111111112);  convert_element_type_default_9 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_3);  add_tensor_2 = mul_tensor_3 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_80);  input_80 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, 768)
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [2], True, dtype = None)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(input_81, input_82, alpha = 1);  input_81 = input_82 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_83);  sub_tensor_2 = input_83 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_5, mul_tensor_7);  mul_tensor_5 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [2], True, dtype = None);  mul_tensor_8 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_7, sum_dim_int_list_4);  sum_dim_int_list_4 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(mul_tensor_6, sum_dim_int_list_3, alpha = 1);  mul_tensor_6 = sum_dim_int_list_3 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(mul_tensor_4, mul_tensor_7);  mul_tensor_7 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1], False, dtype = None);  mul_tensor_10 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], False, dtype = None);  mul_tensor_4 = None
        return (mm_default, convert_element_type_default_2, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_8, mul_tensor_9, sub_tensor_3, sum_dim_int_list_5, sum_dim_int_list_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
