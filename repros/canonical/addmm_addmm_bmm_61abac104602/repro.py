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
    def forward(self, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59):
        embedding_default = torch.ops.aten.embedding.default(input_40, input_41, 0, False, False);  input_40 = input_41 = None
        slice_tensor = torch.ops.aten.slice.Tensor(input_42, 1, 0, 128, 1);  input_42 = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, slice_tensor, alpha = 1);  embedding_default = slice_tensor = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_43, input_44, 0, False, False);  input_43 = input_44 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, embedding_default_1, alpha = 1);  add_tensor = embedding_default_1 = None
        mean_dim = torch.ops.aten.mean.dim(add_tensor_1, [-1], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, mean_dim, alpha = 1);  mean_dim = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_45, sub_tensor);  input_45 = sub_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        var_correction = torch.ops.aten.var.correction(convert_element_type_default, [-1], correction = 1.0, keepdim = True);  convert_element_type_default = None
        sqrt_default = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(sqrt_default, torch.bfloat16);  sqrt_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-06, alpha = 1);  convert_element_type_default_1 = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_2);  mul_tensor = add_tensor_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(div_tensor, input_46, alpha = 1);  div_tensor = input_46 = None
        view_default = torch.ops.aten.view.default(add_tensor_3, input_47)
        permute_default = torch.ops.aten.permute.default(input_48, [1, 0]);  input_48 = None
        addmm_default = torch.ops.aten.addmm.default(input_49, view_default, permute_default, beta = 1, alpha = 1);  input_49 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_50);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_51);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_52, implicit = False);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_3, input_47)
        permute_default_2 = torch.ops.aten.permute.default(input_53, [1, 0]);  input_53 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_54, view_default_3, permute_default_2, beta = 1, alpha = 1);  input_54 = view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_50);  addmm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_51);  view_default_4 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_3, [0, 1, 3, 2]);  permute_default_3 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_4, input_55, implicit = False);  permute_default_4 = input_55 = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_56);  clone_default = input_56 = None
        bmm_default = torch.ops.aten.bmm.default(input_57, view_default_6);  input_57 = view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_3, input_47);  add_tensor_3 = input_47 = None
        permute_default_5 = torch.ops.aten.permute.default(input_58, [1, 0]);  input_58 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_59, view_default_7, permute_default_5, beta = 1, alpha = 1);  input_59 = view_default_7 = permute_default_5 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_2, input_50);  addmm_default_2 = input_50 = None
        view_default_9 = torch.ops.aten.view.default(view_default_8, input_51);  view_default_8 = input_51 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_9, [0, 2, 1, 3]);  view_default_9 = None
        expand_default_2 = torch.ops.aten.expand.default(permute_default_6, input_52, implicit = False);  permute_default_6 = input_52 = None
        return (expand_default, bmm_default, expand_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
