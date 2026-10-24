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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65):
        view_default = torch.ops.aten.view.default(input_44, input_45);  input_44 = input_45 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_46);  view_default = input_46 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_47);  mm_default = input_47 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_48);  view_default_1 = input_48 = None
        slice_tensor = torch.ops.aten.slice.Tensor(view_default_2, 2, 0, 6, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(view_default_2, 2, 6, 12, 1);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(slice_tensor_1, input_49);  slice_tensor_1 = input_49 = None
        permute_default = torch.ops.aten.permute.default(slice_tensor, [0, 2, 1, 3]);  slice_tensor = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        clone_default_1 = torch.ops.aten.clone.default(view_default_3, memory_format = torch.contiguous_format);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(clone_default_1, input_50);  clone_default_1 = input_50 = None
        bmm_default = torch.ops.aten.bmm.default(input_51, view_default_4);  input_51 = None
        add_tensor = torch.ops.aten.add.Tensor(input_52, input_53, alpha = 1);  input_52 = input_53 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor, input_54);  add_tensor = input_54 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_55, alpha = 1);  convert_element_type_default = input_55 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_56);  exp_default = input_56 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16)
        expand_default = torch.ops.aten.expand.default(convert_element_type_default_1, input_57, implicit = False);  convert_element_type_default_1 = input_57 = None
        permute_default_1 = torch.ops.aten.permute.default(expand_default, [0, 2, 1]);  expand_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(permute_default_1, torch.float32);  permute_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_2, convert_element_type_default_3);  convert_element_type_default_2 = convert_element_type_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(bmm_default, torch.float32);  bmm_default = None
        view_default_6 = torch.ops.aten.view.default(convert_element_type_default_4, input_58);  convert_element_type_default_4 = input_58 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_59);  view_default_6 = input_59 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1]);  view_default_7 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(permute_default_2, torch.float32);  permute_default_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_5, div_tensor);  convert_element_type_default_5 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_1);  neg_default = sum_dim_int_list = mul_tensor_1 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_7, input_60);  convert_element_type_default_7 = input_60 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_8, [0, 1], True, dtype = torch.float32)
        view_default_9 = torch.ops.aten.view.default(sum_dim_int_list_1, input_61);  sum_dim_int_list_1 = input_61 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_9, torch.bfloat16);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(view_default_8, input_62);  view_default_8 = input_62 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_10, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default_3, input_63);  permute_default_3 = input_63 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_10, input_64);  view_default_10 = input_64 = None
        view_default_11 = torch.ops.aten.view.default(mm_default_2, input_65);  mm_default_2 = input_65 = None
        return (clone_default, convert_element_type_default_6, convert_element_type_default_8, mm_default_1, view_default_11)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
