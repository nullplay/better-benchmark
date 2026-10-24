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
        embedding_default = torch.ops.aten.embedding.default(input_39, input_40, 199999, False, False);  input_39 = input_40 = None
        view_default = torch.ops.aten.view.default(input_41, input_42);  input_41 = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, view_default, alpha = 1);  embedding_default = view_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(input_43, [1], False, dtype = None);  input_43 = None
        view_default_1 = torch.ops.aten.view.default(sum_dim_int_list, input_42);  sum_dim_int_list = input_42 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_1, alpha = 1);  add_tensor = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32);  add_tensor_1 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mean_dim, 1e-05, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_44, mul_tensor);  input_44 = mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_1, input_45)
        permute_default = torch.ops.aten.permute.default(input_46, [1, 0]);  input_46 = None
        addmm_default = torch.ops.aten.addmm.default(input_47, view_default_2, permute_default, beta = 1, alpha = 1);  input_47 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default, input_48);  addmm_default = input_48 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_49);  view_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        split_tensor = torch.ops.aten.split.Tensor(permute_default_1, 32, -1);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_1, input_45)
        permute_default_2 = torch.ops.aten.permute.default(input_50, [1, 0]);  input_50 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_51, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_51 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_52);  addmm_default_1 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_49);  view_default_6 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        split_tensor_1 = torch.ops.aten.split.Tensor(permute_default_3, 32, -1);  permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_1, input_45);  convert_element_type_default_1 = input_45 = None
        permute_default_4 = torch.ops.aten.permute.default(input_53, [1, 0]);  input_53 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_54, view_default_8, permute_default_4, beta = 1, alpha = 1);  input_54 = view_default_8 = permute_default_4 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default_2, input_52);  addmm_default_2 = input_52 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_49);  view_default_9 = input_49 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_10, [0, 2, 1, 3]);  view_default_10 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(permute_default_5, 2)
        expand_default = torch.ops.aten.expand.default(unsqueeze_default, input_55, implicit = False);  unsqueeze_default = input_55 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_11 = torch.ops.aten.view.default(clone_default, input_56);  clone_default = None
        expand_default_1 = torch.ops.aten.expand.default(view_default_11, input_56, implicit = False);  view_default_11 = input_56 = None
        return (split_tensor, split_tensor_1, permute_default_5, expand_default_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
