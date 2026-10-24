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
    def forward(self, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79):
        add_tensor = torch.ops.aten.add.Tensor(input_53, input_54, alpha = 1);  input_53 = input_54 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, input_55);  add_tensor = input_55 = None
        view_default = torch.ops.aten.view.default(input_56, input_57);  input_56 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default, alpha = 1);  mul_tensor = convert_element_type_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_1, input_58);  add_tensor_1 = input_58 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16)
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_59);  convert_element_type_default_1 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_60);  view_default_1 = input_60 = None
        view_default_2 = torch.ops.aten.view.default(input_61, input_62);  input_61 = input_62 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_57);  clone_default = input_57 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_59);  view_default_3 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_4, input_63);  input_63 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [1, 0]);  view_default_4 = None
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_64);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_1, input_65);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_66, convert_element_type_default_2, alpha = 1);  input_66 = convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_67, torch.float32);  input_67 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_68, torch.float32);  input_68 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(convert_element_type_default_4, convert_element_type_default_5, alpha = 1);  convert_element_type_default_4 = convert_element_type_default_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_3, input_69);  add_tensor_3 = input_69 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_6, input_59);  convert_element_type_default_6 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_6, input_70);  input_70 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [1, 0]);  view_default_6 = None
        mm_default_4 = torch.ops.aten.mm.default(permute_default_2, input_64);  permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_3, input_65);  mm_default_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_7, alpha = 1);  add_tensor_2 = convert_element_type_default_7 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_71);  mul_tensor_1 = input_71 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_9, input_59);  convert_element_type_default_9 = input_59 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_8, input_72);  input_72 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_8, [1, 0]);  view_default_8 = None
        mm_default_6 = torch.ops.aten.mm.default(permute_default_3, input_64);  permute_default_3 = input_64 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_5, input_65);  mm_default_5 = input_65 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_10, alpha = 1);  add_tensor_4 = convert_element_type_default_10 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mm_default_6, torch.float32);  mm_default_6 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor_5, [0, 1], True, dtype = torch.float32)
        view_default_10 = torch.ops.aten.view.default(sum_dim_int_list, input_73);  sum_dim_int_list = input_73 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(input_74, input_75, alpha = 1);  input_74 = input_75 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_5, add_tensor_6);  add_tensor_6 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_5, input_76);  add_tensor_5 = input_76 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1], True, dtype = torch.float32);  mul_tensor_4 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0], True, dtype = torch.float32)
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(input_77, input_78, [input_79], mul_tensor_5);  input_77 = input_78 = input_79 = mul_tensor_5 = None
        return (mm_default, convert_element_type_default_3, convert_element_type_default_8, convert_element_type_default_11, view_default_10, sum_dim_int_list_1, convert_element_type_default_12, sum_dim_int_list_2, _unsafe_masked_index_put_accumulate_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
