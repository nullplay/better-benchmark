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
    def forward(self, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        slice_scatter_default = torch.ops.aten.slice_scatter.default(input_24, input_25, 1, 1, 9223372036854775807, 1);  input_24 = input_25 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_26, input_27);  input_26 = input_27 = None
        add_tensor = torch.ops.aten.add.Tensor(slice_scatter_default, mul_tensor, alpha = 1);  slice_scatter_default = mul_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_28, input_29, alpha = 1);  input_28 = input_29 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_30, sub_tensor);  input_30 = sub_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor_1, alpha = 1);  add_tensor = mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_31, input_32);  input_31 = input_32 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(input_33, input_34, alpha = 1);  input_33 = input_34 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_2, alpha = 1);  sub_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_35, sub_tensor_2);  input_35 = sub_tensor_2 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_3, alpha = 1);  add_tensor_1 = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_2, input_36);  add_tensor_2 = input_36 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_37);  convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(view_default, input_38);  input_38 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        permute_default_1 = torch.ops.aten.permute.default(input_39, [0, 2, 1, 3]);  input_39 = None
        view_default_1 = torch.ops.aten.view.default(permute_default_1, input_40);  permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_37);  view_default_1 = input_37 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, view_default_2);  permute_default = view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_40);  mm_default = input_40 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_41);  view_default_3 = input_41 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        return (convert_element_type_default_1, permute_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
