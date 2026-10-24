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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48):
        embedding_default = torch.ops.aten.embedding.default(input_35, input_36, -1, False, False);  input_35 = input_36 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(embedding_default, torch.float32);  embedding_default = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, rsqrt_default);  convert_element_type_default = rsqrt_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_37, convert_element_type_default_1);  input_37 = convert_element_type_default_1 = None
        view_default = torch.ops.aten.view.default(mul_tensor_1, input_38)
        permute_default = torch.ops.aten.permute.default(input_39, [1, 0]);  input_39 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_40);  mm_default = input_40 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_41);  view_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_2, 2)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, input_42);  convert_element_type_default_2 = input_42 = None
        view_default_3 = torch.ops.aten.view.default(mul_tensor_1, input_38)
        permute_default_1 = torch.ops.aten.permute.default(input_43, [1, 0]);  input_43 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_3, permute_default_1);  view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_1, input_44);  mm_default_1 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_41);  view_default_4 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        pow_tensor_scalar_2 = torch.ops.aten.pow.Tensor_Scalar(convert_element_type_default_3, 2)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_45);  convert_element_type_default_3 = input_45 = None
        view_default_6 = torch.ops.aten.view.default(mul_tensor_1, input_38);  mul_tensor_1 = input_38 = None
        permute_default_2 = torch.ops.aten.permute.default(input_46, [1, 0]);  input_46 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_6, permute_default_2);  view_default_6 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(mm_default_2, input_44);  mm_default_2 = input_44 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_41);  view_default_7 = input_41 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(permute_default_3, 2)
        expand_default = torch.ops.aten.expand.default(unsqueeze_default, input_47, implicit = False);  unsqueeze_default = input_47 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_9 = torch.ops.aten.view.default(clone_default, input_48);  clone_default = input_48 = None
        return (pow_tensor_scalar_1, mul_tensor_2, pow_tensor_scalar_2, mul_tensor_3, permute_default_3, view_default_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
