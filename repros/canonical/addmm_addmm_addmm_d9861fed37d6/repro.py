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
    def forward(self, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        slice_tensor = torch.ops.aten.slice.Tensor(input_31, 1, 0, 128, 1);  input_31 = None
        full_default = torch.ops.aten.full.default(input_32, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_32 = None
        embedding_default = torch.ops.aten.embedding.default(input_33, input_34, 0, False, False);  input_33 = input_34 = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(embedding_default, 1, 1, 9223372036854775807, 1)
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(slice_tensor_1, [0, 0, 0, 1, 0, 0], 0.0);  slice_tensor_1 = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(embedding_default, 1, 0, -1, 1)
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(slice_tensor_2, [0, 0, 1, 0, 0, 0], 0.0);  slice_tensor_2 = None
        cat_default = torch.ops.aten.cat.default([constant_pad_nd_default, embedding_default, constant_pad_nd_default_1], 2);  constant_pad_nd_default = embedding_default = constant_pad_nd_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_35, torch.bfloat16);  input_35 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_36, torch.bfloat16);  input_36 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(cat_default, torch.bfloat16);  cat_default = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_37);  convert_element_type_default_2 = input_37 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_38)
        embedding_default_1 = torch.ops.aten.embedding.default(input_39, slice_tensor, -1, False, False);  input_39 = slice_tensor = None
        embedding_default_2 = torch.ops.aten.embedding.default(input_40, full_default, -1, False, False);  input_40 = full_default = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_1, embedding_default_1, alpha = 1);  view_default_1 = embedding_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2, alpha = 1);  add_tensor = embedding_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_1, input_41);  add_tensor_1 = input_41 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor, input_42, alpha = 1);  mul_tensor = input_42 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16)
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_3, input_43);  convert_element_type_default_3 = input_43 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_44, view_default_2, input_45, beta = 1, alpha = 1);  input_44 = input_45 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_46, view_default_2, input_47, beta = 1, alpha = 1);  input_46 = input_47 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_48, view_default_2, input_49, beta = 1, alpha = 1);  input_48 = input_49 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_50, input_51, input_52, beta = 1, alpha = 1);  input_50 = input_51 = input_52 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_4, input_38);  addmm_default_4 = input_38 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_3, add_tensor_2, alpha = 1);  view_default_3 = add_tensor_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_3, input_53);  input_53 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_1, input_54, alpha = 1);  mul_tensor_1 = input_54 = None
        return (addmm_default, view_default_2, addmm_default_1, addmm_default_2, addmm_default_3, add_tensor_3, add_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
