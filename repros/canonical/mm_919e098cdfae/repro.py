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
    def forward(self, input_18, input_19, input_20, input_21, input_22):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_18, torch.bfloat16);  input_18 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_19, torch.bfloat16);  input_19 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_20);  convert_element_type_default_1 = input_20 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 3, 0, 0], 0);  permute_default = None
        mm_default = torch.ops.aten.mm.default(view_default, constant_pad_nd_default);  view_default = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -3, 1);  mm_default = None
        view_default_1 = torch.ops.aten.view.default(slice_tensor, input_21);  slice_tensor = input_21 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_22);  input_22 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_2, [1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default, alpha = 1);  convert_element_type_default_2 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, log_default, alpha = 1);  sub_tensor = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(convert_element_type_default_3, torch.float32);  convert_element_type_default_3 = None
        return (view_default_1, amax_default, log_default, convert_element_type_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
