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
    def forward(self, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        mul_tensor = torch.ops.aten.mul.Tensor(input_36, input_37);  input_36 = input_37 = None
        view_default = torch.ops.aten.view.default(mul_tensor, input_38);  mul_tensor = None
        mm_default = torch.ops.aten.mm.default(view_default, input_39);  view_default = input_39 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_40);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_41);  view_default_1 = input_41 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_42);  clone_default = input_42 = None
        bmm_default = torch.ops.aten.bmm.default(input_43, view_default_3);  input_43 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_44);  view_default_3 = input_44 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(bmm_default_1, torch.float32);  bmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(input_45, input_46);  input_45 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_4, input_47, alpha = 1);  view_default_4 = input_47 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor, input_48);  add_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(view_default_5, input_49, alpha = 1);  view_default_5 = input_49 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_50);  exp_default = input_50 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, div_tensor);  convert_element_type_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_1);  neg_default = sum_dim_int_list = mul_tensor_1 = None
        view_default_6 = torch.ops.aten.view.default(fma_default, input_46);  fma_default = input_46 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_6, torch.bfloat16);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_1, input_48);  convert_element_type_default_1 = input_48 = None
        bmm_default_2 = torch.ops.aten.bmm.default(input_51, view_default_7);  input_51 = None
        bmm_default_3 = torch.ops.aten.bmm.default(view_default_7, input_52);  view_default_7 = input_52 = None
        permute_default_1 = torch.ops.aten.permute.default(bmm_default_2, [0, 2, 1]);  bmm_default_2 = None
        view_default_8 = torch.ops.aten.view.default(bmm_default, input_53);  bmm_default = None
        view_default_9 = torch.ops.aten.view.default(bmm_default_3, input_53);  bmm_default_3 = input_53 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_10 = torch.ops.aten.view.default(clone_default_1, input_40);  clone_default_1 = input_40 = None
        view_default_11 = torch.ops.aten.view.default(view_default_10, input_38);  view_default_10 = input_38 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_11, input_54);  input_54 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_11, [1, 0])
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_11, [0], True, dtype = torch.float32);  view_default_11 = None
        return (permute_default_1, view_default_9, mm_default_1, permute_default_3, sum_dim_int_list_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
