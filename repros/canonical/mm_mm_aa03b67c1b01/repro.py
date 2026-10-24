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
    def forward(self, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31):
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_20, 1);  input_20 = None
        expand_default = torch.ops.aten.expand.default(unsqueeze_default, input_21, implicit = False);  unsqueeze_default = input_21 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 196);  expand_default = None
        full_default = torch.ops.aten.full.default(input_22, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_22 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(full_default, div_scalar, 1, 1, 9223372036854775807, 1);  full_default = div_scalar = None
        mul_tensor = torch.ops.aten.mul.Tensor(slice_scatter_default, input_23);  slice_scatter_default = input_23 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default, input_24);  convert_element_type_default = input_24 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_25);  input_25 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_26);  permute_default = input_26 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_27);  mm_default = input_27 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_28, 0.5);  input_28 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_29, input_30);  input_29 = input_30 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2, alpha = 1);  mul_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, add_tensor);  convert_element_type_default_2 = add_tensor = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_3, input_31);  convert_element_type_default_3 = input_31 = None
        return (convert_element_type_default_1, view_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
