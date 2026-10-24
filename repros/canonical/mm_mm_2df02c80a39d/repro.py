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
    def forward(self, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31):
        mm_default = torch.ops.aten.mm.default(input_18, input_19);  input_19 = None
        permute_default = torch.ops.aten.permute.default(input_18, [1, 0]);  input_18 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_20);  permute_default = input_20 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(mm_default, input_21);  mm_default = input_21 = None
        full_default = torch.ops.aten.full.default(input_22, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_22 = None
        where_self = torch.ops.aten.where.self(input_23, full_default, view_default);  input_23 = full_default = view_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(where_self, [0, 2, 3], False, dtype = None)
        convolution_backward_default = torch.ops.aten.convolution_backward.default(where_self, input_24, input_25, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_self = input_24 = input_25 = None
        getitem = convolution_backward_default[0]
        getitem_1 = convolution_backward_default[1];  convolution_backward_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(sum_dim_int_list, torch.float32);  sum_dim_int_list = None
        squeeze_dim = torch.ops.aten.squeeze.dim(getitem, 3);  getitem = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(squeeze_dim, 2);  squeeze_dim = None
        full_default_1 = torch.ops.aten.full.default(input_26, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_26 = None
        as_strided_scatter_default = torch.ops.aten.as_strided_scatter.default(full_default_1, squeeze_dim_1, input_27, input_28, input_29);  full_default_1 = squeeze_dim_1 = input_27 = input_28 = None
        as_strided_default = torch.ops.aten.as_strided.default(as_strided_scatter_default, input_30, input_31, input_29);  as_strided_scatter_default = input_30 = input_31 = input_29 = None
        return (convert_element_type_default, convert_element_type_default_1, convert_element_type_default_2, as_strided_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
