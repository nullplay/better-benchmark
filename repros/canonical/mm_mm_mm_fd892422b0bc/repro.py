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
    def forward(self, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86):
        mul_tensor = torch.ops.aten.mul.Tensor(input_62, input_63);  input_62 = input_63 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 512)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        view_default = torch.ops.aten.view.default(input_64, input_65);  input_64 = None
        relu_default = torch.ops.aten.relu.default(view_default);  view_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(relu_default, torch.float32)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_66, alpha = 1);  convert_element_type_default = input_66 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_67);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4, alpha = 1);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor = torch.ops.aten.div.Tensor(input_67, 512);  input_67 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default = torch.ops.aten.full.default(input_68, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_68 = None
        where_self = torch.ops.aten.where.self(le_scalar, full_default, convert_element_type_default_1);  le_scalar = full_default = convert_element_type_default_1 = None
        view_default_1 = torch.ops.aten.view.default(where_self, input_69);  where_self = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_70);  input_70 = None
        permute_default = torch.ops.aten.permute.default(view_default_1, [1, 0]);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_71);  permute_default = input_71 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_65);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 1], True, dtype = torch.float32)
        view_default_3 = torch.ops.aten.view.default(sum_dim_int_list_2, input_72);  sum_dim_int_list_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_73);  input_73 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_74);  convert_element_type_default_2 = input_74 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1], True, dtype = torch.float32);  mul_tensor_6 = None
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list_3, input_72);  sum_dim_int_list_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16)
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_4, input_69);  convert_element_type_default_4 = input_69 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_5, input_75);  input_75 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_5, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_76);  permute_default_1 = input_76 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(view_default_5, [0], True, dtype = torch.float32);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list_4, input_72);  sum_dim_int_list_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_6, torch.bfloat16);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_2, input_77);  mm_default_2 = input_77 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        view_default_8 = torch.ops.aten.view.default(input_78, input_79);  input_78 = input_79 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_8, input_80);  view_default_8 = input_80 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_4, input_65);  mm_default_4 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_7, convert_element_type_default_8, alpha = 1);  mul_tensor_7 = convert_element_type_default_8 = None
        mm_default_5 = torch.ops.aten.mm.default(input_81, input_82);  input_81 = input_82 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_5, input_65);  mm_default_5 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_9, alpha = 1);  add_tensor = convert_element_type_default_9 = None
        view_default_11 = torch.ops.aten.view.default(input_83, input_65);  input_83 = input_65 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32);  view_default_11 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_10, alpha = 1);  add_tensor_1 = convert_element_type_default_10 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(add_tensor_2, [0, 1], True, dtype = torch.float32)
        view_default_12 = torch.ops.aten.view.default(sum_dim_int_list_5, input_72);  sum_dim_int_list_5 = input_72 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(add_tensor_2, input_84);  input_84 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(add_tensor_2, input_85);  add_tensor_2 = input_85 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1], True, dtype = torch.float32);  mul_tensor_8 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16)
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_9, input_86, alpha = 1);  mul_tensor_9 = input_86 = None
        return (convert_element_type_default_3, view_default_3, view_default_4, convert_element_type_default_5, convert_element_type_default_6, convert_element_type_default_7, view_default_12, sum_dim_int_list_6, convert_element_type_default_11, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
