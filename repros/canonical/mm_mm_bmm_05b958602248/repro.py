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
    def forward(self, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71):
        mul_tensor = torch.ops.aten.mul.Tensor(input_48, input_49);  input_48 = input_49 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 256)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(input_50, input_51, alpha = 1);  input_50 = input_51 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, input_52);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor, mul_tensor_2);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, sum_dim_int_list_1);  mul_tensor_2 = sum_dim_int_list_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_4, alpha = 1);  sub_tensor_1 = mul_tensor_4 = None
        div_tensor = torch.ops.aten.div.Tensor(input_52, 256);  input_52 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_5, torch.bfloat16);  mul_tensor_5 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_53, torch.bfloat16);  input_53 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_6);  mul_tensor_6 = None
        view_default = torch.ops.aten.view.default(mul_tensor_7, input_54);  mul_tensor_7 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_55);  input_55 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_56);  permute_default = input_56 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_57);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_58);  view_default_1 = input_58 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_59);  clone_default = input_59 = None
        bmm_default = torch.ops.aten.bmm.default(input_60, view_default_3);  input_60 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_61);  view_default_3 = input_61 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_62);  bmm_default = input_62 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default_1, input_63);  bmm_default_1 = input_63 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_default_5, input_64);  view_default_5 = input_64 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        view_default_6 = torch.ops.aten.view.default(input_65, input_57);  input_65 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_54);  view_default_6 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_7, input_66);  view_default_7 = input_66 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_2, input_57);  mm_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default, view_default_8, alpha = 1);  convert_element_type_default = view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(input_67, input_54);  input_67 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_9, input_68);  view_default_9 = input_68 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_3, input_57);  mm_default_3 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_10, alpha = 1);  add_tensor = view_default_10 = None
        mm_default_4 = torch.ops.aten.mm.default(input_69, input_70);  input_69 = input_70 = None
        view_default_11 = torch.ops.aten.view.default(mm_default_4, input_57);  mm_default_4 = input_57 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_11, alpha = 1);  add_tensor_1 = view_default_11 = None
        view_default_12 = torch.ops.aten.view.default(add_tensor_2, input_54);  add_tensor_2 = input_54 = None
        mm_default_5 = torch.ops.aten.mm.default(view_default_12, input_71);  input_71 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_12, [1, 0])
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_12, [0], True, dtype = torch.float32);  view_default_12 = None
        return (convert_element_type_default_2, mul_tensor_8, permute_default_2, mm_default_5, permute_default_3, sum_dim_int_list_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
