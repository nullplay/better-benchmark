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
    def forward(self, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43):
        mul_tensor = torch.ops.aten.mul.Tensor(input_26, input_27);  input_26 = input_27 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, input_28, alpha = 1);  mul_tensor = input_28 = None
        view_default = torch.ops.aten.view.default(add_tensor, input_29);  add_tensor = input_29 = None
        permute_default = torch.ops.aten.permute.default(input_30, [1, 0]);  input_30 = None
        addmm_default = torch.ops.aten.addmm.default(input_31, view_default, permute_default, beta = 1, alpha = 1);  input_31 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_32);  addmm_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(view_default_1, input_33);  view_default_1 = input_33 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_34, alpha = 1);  mul_tensor_1 = input_34 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_35)
        permute_default_1 = torch.ops.aten.permute.default(input_36, [1, 0]);  input_36 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_37, view_default_2, permute_default_1, beta = 1, alpha = 1);  input_37 = view_default_2 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_32);  addmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_38);  view_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        mul_scalar = torch.ops.aten.mul.Scalar(permute_default_2, 0.4204482076268573);  permute_default_2 = None
        expand_default = torch.ops.aten.expand.default(mul_scalar, input_39, implicit = False);  mul_scalar = input_39 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_40);  clone_default = input_40 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_1, input_35);  add_tensor_1 = input_35 = None
        permute_default_3 = torch.ops.aten.permute.default(input_41, [1, 0]);  input_41 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_42, view_default_6, permute_default_3, beta = 1, alpha = 1);  input_42 = view_default_6 = permute_default_3 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_2, input_32);  addmm_default_2 = input_32 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_38);  view_default_7 = input_38 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 1, 3, 2]);  permute_default_4 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_5, input_43);  view_default_5 = input_43 = None
        return (permute_default_5, bmm_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
