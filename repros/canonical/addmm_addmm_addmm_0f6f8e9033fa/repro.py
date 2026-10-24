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
    def forward(self, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65):
        embedding_default = torch.ops.aten.embedding.default(input_39, input_40, 0, False, False);  input_40 = None
        slice_tensor = torch.ops.aten.slice.Tensor(embedding_default, 1, 1, 9223372036854775807, 1)
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, 0, 0, 1, 0, 0], 0.0);  slice_tensor = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(embedding_default, 1, 0, -1, 1)
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(slice_tensor_1, [0, 0, 1, 0, 0, 0], 0.0);  slice_tensor_1 = None
        cat_default = torch.ops.aten.cat.default([constant_pad_nd_default, embedding_default, constant_pad_nd_default_1], 2);  constant_pad_nd_default = embedding_default = constant_pad_nd_default_1 = None
        view_default = torch.ops.aten.view.default(cat_default, input_41);  cat_default = input_41 = None
        permute_default = torch.ops.aten.permute.default(input_42, [1, 0]);  input_42 = None
        addmm_default = torch.ops.aten.addmm.default(input_43, view_default, permute_default, beta = 1, alpha = 1);  input_43 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_44);  addmm_default = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(input_45, 1, 0, 128, 1);  input_45 = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_46, slice_tensor_2, -1, False, False);  input_46 = slice_tensor_2 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_1, embedding_default_1, alpha = 1);  view_default_1 = embedding_default_1 = None
        full_default = torch.ops.aten.full.default(input_47, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_47 = None
        embedding_default_2 = torch.ops.aten.embedding.default(input_48, full_default, -1, False, False);  input_48 = full_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2, alpha = 1);  add_tensor = embedding_default_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_1, input_49);  add_tensor_1 = input_49 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor, input_50, alpha = 1);  mul_tensor = input_50 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_2, input_51)
        permute_default_1 = torch.ops.aten.permute.default(input_52, [1, 0]);  input_52 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_53, view_default_2, permute_default_1, beta = 1, alpha = 1);  input_53 = view_default_2 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_54);  addmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(add_tensor_2, input_51)
        permute_default_2 = torch.ops.aten.permute.default(input_55, [1, 0]);  input_55 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_56, view_default_4, permute_default_2, beta = 1, alpha = 1);  input_56 = view_default_4 = permute_default_2 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_54);  addmm_default_2 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_2, input_51);  input_51 = None
        permute_default_3 = torch.ops.aten.permute.default(input_57, [1, 0]);  input_57 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_58, view_default_6, permute_default_3, beta = 1, alpha = 1);  input_58 = view_default_6 = permute_default_3 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_3, input_54);  addmm_default_3 = input_54 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_59, input_60, input_61, beta = 1, alpha = 1);  input_59 = input_60 = input_61 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_4, input_44);  addmm_default_4 = input_44 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_8, add_tensor_2, alpha = 1);  view_default_8 = add_tensor_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_3, input_62);  add_tensor_3 = input_62 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_1, input_63, alpha = 1);  mul_tensor_1 = input_63 = None
        permute_default_4 = torch.ops.aten.permute.default(input_39, [1, 0]);  input_39 = None
        cat_default_1 = torch.ops.aten.cat.default([permute_default_4, input_64], 0);  permute_default_4 = input_64 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(cat_default_1, [0, 6, 0, 0], 0);  cat_default_1 = None
        mm_default = torch.ops.aten.mm.default(input_65, constant_pad_nd_default_2);  input_65 = constant_pad_nd_default_2 = None
        return (view_default_3, view_default_5, view_default_7, add_tensor_4, mm_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
