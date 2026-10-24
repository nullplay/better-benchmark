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
    def forward(self, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34):
        add_tensor = torch.ops.aten.add.Tensor(input_21, input_22, alpha = 1);  input_21 = input_22 = None
        bmm_default = torch.ops.aten.bmm.default(input_23, input_24);  input_23 = input_24 = None
        view_default = torch.ops.aten.view.default(bmm_default, input_25);  bmm_default = input_25 = None
        permute_default = torch.ops.aten.permute.default(view_default, [0, 1, 3, 2]);  view_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_26, torch.float32);  input_26 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default, alpha = 1);  add_tensor = convert_element_type_default = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_1 = torch.ops.aten.view.default(permute_default_1, input_27);  permute_default_1 = input_27 = None
        clone_default = torch.ops.aten.clone.default(view_default_1, memory_format = torch.contiguous_format);  view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_28);  clone_default = input_28 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_2, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default_2, input_29);  permute_default_2 = input_29 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, input_30);  view_default_2 = input_30 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_31);  mm_default_1 = input_31 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_1, alpha = 1);  add_tensor_1 = convert_element_type_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_32, 1.1111111111111112);  input_32 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor);  add_tensor_2 = mul_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_33);  input_33 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_34);  mul_tensor_1 = input_34 = None
        return (convert_element_type_default_2, mul_tensor_2, mul_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
