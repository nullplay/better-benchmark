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
    def forward(self, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32):
        permute_default = torch.ops.aten.permute.default(input_21, [0, 1, 4, 2, 3, 5]);  input_21 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default = torch.ops.aten.view.default(clone_default, input_22);  clone_default = input_22 = None
        view_default_1 = torch.ops.aten.view.default(view_default, input_23);  view_default = input_23 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.bfloat16);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_24, torch.bfloat16);  input_24 = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_1, 0.9956521736457944);  convert_element_type_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, div_tensor);  convert_element_type_default = div_tensor = None
        view_default_2 = torch.ops.aten.view.default(mul_tensor, input_25);  mul_tensor = input_25 = None
        mm_default = torch.ops.aten.mm.default(view_default_2, input_26);  input_26 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [1, 0]);  view_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_1, input_27);  permute_default_1 = input_27 = None
        view_default_3 = torch.ops.aten.view.default(mm_default, input_28);  mm_default = input_28 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_29, 0.5);  input_29 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_30, input_31);  input_30 = input_31 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2, alpha = 1);  mul_tensor_1 = mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, add_tensor);  convert_element_type_default_3 = add_tensor = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.bfloat16);  mul_tensor_3 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_4, input_32);  convert_element_type_default_4 = input_32 = None
        return (convert_element_type_default_2, view_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
