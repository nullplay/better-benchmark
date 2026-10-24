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
    def forward(self, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56):
        bmm_default = torch.ops.aten.bmm.default(input_39, input_40);  input_39 = input_40 = None
        view_default = torch.ops.aten.view.default(bmm_default, input_41);  bmm_default = input_41 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_42);  convert_element_type_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(input_42);  input_42 = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_1, 16.0);  convert_element_type_default_1 = None
        view_default_1 = torch.ops.aten.view.default(div_tensor, input_43);  div_tensor = input_43 = None
        bmm_default_1 = torch.ops.aten.bmm.default(input_44, view_default_1);  input_44 = None
        bmm_default_2 = torch.ops.aten.bmm.default(view_default_1, input_45);  view_default_1 = input_45 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default_1, input_46);  bmm_default_1 = input_46 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default_2, input_47);  bmm_default_2 = input_47 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_2, [0, 1, 3, 2]);  convert_element_type_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(permute_default, torch.bfloat16);  permute_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_3, torch.bfloat16);  view_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1, 3]);  convert_element_type_default_4 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_3, [0, 2, 1, 3]);  convert_element_type_default_3 = None
        slice_tensor = torch.ops.aten.slice.Tensor(permute_default_1, 3, 0, 64, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(permute_default_1, 3, 64, 256, 1);  permute_default_1 = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(permute_default_2, 3, 0, 64, 1)
        slice_tensor_3 = torch.ops.aten.slice.Tensor(permute_default_2, 3, 64, 256, 1);  permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(input_48, input_49);  input_48 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(slice_tensor, view_default_4)
        view_default_5 = torch.ops.aten.view.default(mul_tensor_1, input_50);  mul_tensor_1 = None
        view_default_6 = torch.ops.aten.view.default(input_51, input_49);  input_51 = input_49 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(slice_tensor, view_default_6);  slice_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_52, mul_tensor_2, alpha = 1);  input_52 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(slice_tensor_2, view_default_4);  view_default_4 = None
        view_default_7 = torch.ops.aten.view.default(mul_tensor_3, input_50);  mul_tensor_3 = input_50 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(slice_tensor_2, view_default_6);  slice_tensor_2 = view_default_6 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_53, mul_tensor_4, alpha = 1);  input_53 = mul_tensor_4 = None
        full_default = torch.ops.aten.full.default(input_54, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_54 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_1, 3, 64, 9223372036854775807, 1);  slice_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(slice_scatter_default, input_55, alpha = 1);  slice_scatter_default = input_55 = None
        slice_scatter_default_1 = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_3, 3, 64, 9223372036854775807, 1);  full_default = slice_tensor_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(slice_scatter_default_1, input_56, alpha = 1);  slice_scatter_default_1 = input_56 = None
        return (view_default_5, add_tensor, view_default_7, add_tensor_1, add_tensor_2, add_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
