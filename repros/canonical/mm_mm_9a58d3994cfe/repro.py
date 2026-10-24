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
    def forward(self, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_22, torch.float32);  input_22 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_23, torch.float32);  input_23 = None
        sub_tensor = torch.ops.aten.sub.Tensor(1, convert_element_type_default_1, alpha = 1)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, sub_tensor);  convert_element_type_default_1 = sub_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mm_default = torch.ops.aten.mm.default(convert_element_type_default_2, input_24);  input_24 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_25);  permute_default = input_25 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(mm_default, input_26);  mm_default = input_26 = None
        avg_pool2d_backward_default = torch.ops.aten.avg_pool2d_backward.default(view_default, input_27, [4, 4], [], [0, 0], False, True, None);  view_default = None
        le_scalar = torch.ops.aten.le.Scalar(input_27, 0);  input_27 = None
        full_default = torch.ops.aten.full.default(input_28, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_28 = None
        where_self = torch.ops.aten.where.self(le_scalar, full_default, avg_pool2d_backward_default);  le_scalar = avg_pool2d_backward_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(where_self, torch.float32)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 2, 3], False, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_29);  input_29 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, input_30, alpha = 1);  convert_element_type_default_4 = input_30 = None
        getitem = input_31[0];  input_31 = None
        add_tensor = torch.ops.aten.add.Tensor(where_self, getitem, alpha = 1);  where_self = getitem = None
        where_self_1 = torch.ops.aten.where.self(input_32, full_default, add_tensor);  input_32 = full_default = add_tensor = None
        return (convert_element_type_default_3, sum_dim_int_list, mul_tensor_2, sub_tensor_1, where_self_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
