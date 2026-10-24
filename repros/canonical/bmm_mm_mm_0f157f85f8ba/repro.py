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
    def forward(self, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57):
        view_default = torch.ops.aten.view.default(input_39, input_40);  input_39 = None
        bmm_default = torch.ops.aten.bmm.default(input_41, input_42);  input_41 = input_42 = None
        exp_default = torch.ops.aten.exp.default(input_43);  input_43 = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_44);  exp_default = input_44 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(bmm_default, torch.float32);  bmm_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, div_tensor);  convert_element_type_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_45);  convert_element_type_default_1 = input_45 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_1, [0, 1], True, dtype = torch.float32)
        view_default_2 = torch.ops.aten.view.default(sum_dim_int_list_1, input_46);  sum_dim_int_list_1 = input_46 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.bfloat16);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(view_default_1, input_47);  view_default_1 = input_47 = None
        permute_default = torch.ops.aten.permute.default(view_default_3, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_48);  permute_default = input_48 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, input_49);  view_default_3 = input_49 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_40);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(input_50, input_51, alpha = 1);  input_50 = input_51 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_5, [0, 2, 1]);  convert_element_type_default_5 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default_4, permute_default_1);  permute_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_4, view_default);  view_default_4 = view_default = None
        clone_default = torch.ops.aten.clone.default(input_52, memory_format = torch.contiguous_format);  input_52 = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_40);  clone_default = input_40 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, view_default_5, alpha = 1);  mul_tensor_1 = view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_1, input_53);  add_tensor_1 = input_53 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_6, input_54);  input_54 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [1, 0])
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(view_default_6, [0], True, dtype = torch.float32);  view_default_6 = None
        permute_default_3 = torch.ops.aten.permute.default(mul_tensor_2, [0, 2, 1]);  mul_tensor_2 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(permute_default_3, [0, 2], True, dtype = torch.float32)
        view_default_7 = torch.ops.aten.view.default(sum_dim_int_list_3, input_55);  sum_dim_int_list_3 = input_55 = None
        convolution_backward_default = torch.ops.aten.convolution_backward.default(permute_default_3, input_56, input_57, [0], [1], [0], [1], False, [0], 1, [True, True, False]);  permute_default_3 = input_56 = input_57 = None
        getitem = convolution_backward_default[0]
        getitem_1 = convolution_backward_default[1];  convolution_backward_default = None
        return (convert_element_type_default_3, convert_element_type_default_4, mm_default_2, permute_default_2, sum_dim_int_list_2, view_default_7, getitem, getitem_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
