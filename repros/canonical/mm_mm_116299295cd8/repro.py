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
    def forward(self, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30):
        mm_default = torch.ops.aten.mm.default(input_23, input_24);  input_24 = None
        permute_default = torch.ops.aten.permute.default(input_23, [1, 0]);  input_23 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_25);  permute_default = input_25 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(mm_default, input_26);  mm_default = input_26 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_27, torch.float32);  input_27 = None
        le_scalar = torch.ops.aten.le.Scalar(convert_element_type_default_2, -3)
        lt_scalar = torch.ops.aten.lt.Scalar(convert_element_type_default_2, 3)
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_2, 3);  convert_element_type_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(div_tensor, 0.5, alpha = 1);  div_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, add_tensor);  add_tensor = None
        where_self = torch.ops.aten.where.self(lt_scalar, mul_tensor, convert_element_type_default_1);  lt_scalar = mul_tensor = convert_element_type_default_1 = None
        full_default = torch.ops.aten.full.default(input_28, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_28 = None
        where_self_1 = torch.ops.aten.where.self(le_scalar, full_default, where_self);  le_scalar = full_default = where_self = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(where_self_1, torch.bfloat16);  where_self_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_3, [0, 2, 3], False, dtype = None)
        convolution_backward_default = torch.ops.aten.convolution_backward.default(convert_element_type_default_3, input_29, input_30, [1280], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_default_3 = input_29 = input_30 = None
        getitem = convolution_backward_default[0]
        getitem_1 = convolution_backward_default[1];  convolution_backward_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(sum_dim_int_list, torch.float32);  sum_dim_int_list = None
        squeeze_dim = torch.ops.aten.squeeze.dim(getitem, 3);  getitem = None
        return (convert_element_type_default, convert_element_type_default_4, convert_element_type_default_5, squeeze_dim)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
