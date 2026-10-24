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
    def forward(self, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60):
        view_default = torch.ops.aten.view.default(input_40, input_41);  input_40 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_42);  view_default = input_42 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, input_43);  view_default_1 = input_43 = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_41);  mm_default = input_41 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_44);  view_default_2 = input_44 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_45);  clone_default = input_45 = None
        bmm_default = torch.ops.aten.bmm.default(input_46, view_default_4);  input_46 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_4, input_47);  view_default_4 = input_47 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default, input_48);  bmm_default = None
        view_default_6 = torch.ops.aten.view.default(bmm_default_1, input_49);  bmm_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_6, torch.float32);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(input_50, input_49);  input_50 = input_49 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_51, 0);  input_51 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_7, unsqueeze_default, alpha = 1);  view_default_7 = unsqueeze_default = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, input_52, alpha = 1);  add_tensor = input_52 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_53);  exp_default = input_53 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, div_tensor);  convert_element_type_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(fma_default, [0], True, dtype = torch.float32);  fma_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(sum_dim_int_list_1, 0);  sum_dim_int_list_1 = None
        permute_default_1 = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None
        view_default_8 = torch.ops.aten.view.default(permute_default_1, input_54);  permute_default_1 = input_54 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_1, input_55);  convert_element_type_default_1 = input_55 = None
        bmm_default_2 = torch.ops.aten.bmm.default(input_56, view_default_9);  input_56 = None
        bmm_default_3 = torch.ops.aten.bmm.default(view_default_9, input_57);  view_default_9 = input_57 = None
        view_default_10 = torch.ops.aten.view.default(bmm_default_2, input_58);  bmm_default_2 = input_58 = None
        view_default_11 = torch.ops.aten.view.default(bmm_default_3, input_48);  bmm_default_3 = input_48 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_10, [0, 1, 3, 2]);  view_default_10 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default_11, 0.1767766952966369);  view_default_11 = None
        cat_default = torch.ops.aten.cat.default([mul_tensor_1, permute_default_2, view_default_5], 0);  mul_tensor_1 = permute_default_2 = view_default_5 = None
        view_default_12 = torch.ops.aten.view.default(cat_default, input_59);  cat_default = input_59 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_12, [1, 3, 0, 2, 4]);  view_default_12 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_13 = torch.ops.aten.view.default(clone_default_1, input_60);  clone_default_1 = input_60 = None
        return (view_default_8, view_default_13)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
