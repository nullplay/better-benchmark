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
    def forward(self, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43):
        view_default = torch.ops.aten.view.default(input_31, input_32);  input_31 = input_32 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 4, 2, 3]);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 1, 4, 3, 2]);  permute_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(permute_default_1, 4);  permute_default_1 = None
        view_default_1 = torch.ops.aten.view.default(squeeze_dim, input_33);  squeeze_dim = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_1, [1, 2, 0, 4, 3]);  view_default_1 = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_34);  clone_default = input_34 = None
        bmm_default = torch.ops.aten.bmm.default(input_35, view_default_2);  input_35 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_2, input_36);  view_default_2 = input_36 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default, input_37);  bmm_default = input_37 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_3, [4, 0, 1, 3, 2]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default_1, input_38);  bmm_default_1 = input_38 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_3, [4, 1, 2, 3, 0]);  permute_default_3 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(permute_default_4, 4);  permute_default_4 = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(view_default_4, 4);  view_default_4 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(squeeze_dim_2, torch.float32);  squeeze_dim_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_39, torch.float32);  input_39 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        where_self = torch.ops.aten.where.self(input_40, input_41, input_42);  input_40 = input_41 = input_42 = None
        exp_default = torch.ops.aten.exp.default(where_self);  where_self = None
        div_tensor = torch.ops.aten.div.Tensor(exp_default, input_43);  exp_default = input_43 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, div_tensor);  mul_tensor_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [3], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_2);  neg_default = sum_dim_int_list = mul_tensor_2 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.125);  convert_element_type_default_2 = None
        view_default_5 = torch.ops.aten.view.default(squeeze_dim_1, input_33);  squeeze_dim_1 = input_33 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_5, [0, 1, 4, 2, 3]);  view_default_5 = None
        return (mul_tensor_3, permute_default_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
