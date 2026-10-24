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
    def forward(self, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_19, torch.bfloat16);  input_19 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_20, torch.bfloat16);  input_20 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_21);  convert_element_type_default_1 = input_21 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_22);  mm_default = input_22 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32)
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_2, input_23);  convert_element_type_default_2 = input_23 = None
        amax_default = torch.ops.aten.amax.default(view_default_2, [1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(view_default_2, amax_default, alpha = 1);  view_default_2 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, log_default, alpha = 1);  sub_tensor = None
        where_self = torch.ops.aten.where.self(input_24, input_25, input_26);  input_24 = input_25 = input_26 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default = torch.ops.aten.gather.default(sub_tensor_1, 1, unsqueeze_default, sparse_grad = False);  sub_tensor_1 = unsqueeze_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        return (view_default_1, amax_default, log_default, neg_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
