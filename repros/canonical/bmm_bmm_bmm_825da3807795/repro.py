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
    def forward(self, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68):
        bmm_default = torch.ops.aten.bmm.default(input_50, input_51);  input_50 = input_51 = None
        view_default = torch.ops.aten.view.default(bmm_default, input_52);  bmm_default = input_52 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, input_53);  convert_element_type_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(input_53);  input_53 = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_1, 16.0);  convert_element_type_default_1 = None
        view_default_1 = torch.ops.aten.view.default(div_tensor, input_54);  div_tensor = input_54 = None
        bmm_default_1 = torch.ops.aten.bmm.default(input_55, view_default_1);  input_55 = None
        bmm_default_2 = torch.ops.aten.bmm.default(view_default_1, input_56);  view_default_1 = input_56 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default_1, input_57);  bmm_default_1 = input_57 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default_2, input_58);  bmm_default_2 = input_58 = None
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
        expand_default = torch.ops.aten.expand.default(input_59, input_60, implicit = False);  input_59 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_61);  clone_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(slice_tensor, view_default_4)
        view_default_5 = torch.ops.aten.view.default(mul_tensor_1, input_62);  mul_tensor_1 = None
        select_int = torch.ops.aten.select.int(view_default_5, -1, 0)
        select_int_1 = torch.ops.aten.select.int(view_default_5, -1, 1);  view_default_5 = None
        neg_default_1 = torch.ops.aten.neg.default(select_int);  select_int = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(input_63, neg_default_1, 3, 1, 9223372036854775807, 2);  neg_default_1 = None
        slice_scatter_default_1 = torch.ops.aten.slice_scatter.default(input_63, select_int_1, 3, 0, 9223372036854775807, 2);  input_63 = select_int_1 = None
        add_tensor = torch.ops.aten.add.Tensor(slice_scatter_default, slice_scatter_default_1, alpha = 1);  slice_scatter_default = slice_scatter_default_1 = None
        expand_default_1 = torch.ops.aten.expand.default(input_64, input_60, implicit = False);  input_64 = input_60 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_61);  clone_default_1 = input_61 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(slice_tensor, view_default_6);  slice_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_2, alpha = 1);  add_tensor = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(slice_tensor_2, view_default_4);  view_default_4 = None
        view_default_7 = torch.ops.aten.view.default(mul_tensor_3, input_62);  mul_tensor_3 = input_62 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(slice_tensor_2, view_default_6);  slice_tensor_2 = view_default_6 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_65, mul_tensor_4, alpha = 1);  input_65 = mul_tensor_4 = None
        full_default = torch.ops.aten.full.default(input_66, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_66 = None
        slice_scatter_default_2 = torch.ops.aten.slice_scatter.default(full_default, slice_tensor_1, 3, 64, 9223372036854775807, 1);  slice_tensor_1 = None
        slice_scatter_default_3 = torch.ops.aten.slice_scatter.default(full_default, add_tensor_1, 3, 0, 64, 1);  full_default = add_tensor_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(slice_scatter_default_2, slice_scatter_default_3, alpha = 1);  slice_scatter_default_2 = slice_scatter_default_3 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_3, input_67);  add_tensor_3 = input_67 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_68);  view_default_8 = input_68 = None
        return (slice_tensor_3, view_default_7, add_tensor_2, view_default_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
