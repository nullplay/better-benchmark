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
    def forward(self, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52):
        add_tensor = torch.ops.aten.add.Tensor(input_29, input_30, alpha = 1);  input_29 = input_30 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, input_31);  add_tensor = input_31 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, input_32, alpha = 1);  mul_tensor = input_32 = None
        view_default = torch.ops.aten.view.default(add_tensor_1, input_33)
        addmm_default = torch.ops.aten.addmm.default(input_34, view_default, input_35, beta = 1, alpha = 1);  input_34 = view_default = input_35 = None
        view_default_1 = torch.ops.aten.view.default(input_36, input_37);  input_36 = None
        permute_default = torch.ops.aten.permute.default(input_38, [1, 0]);  input_38 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_39, view_default_1, permute_default, beta = 1, alpha = 1);  input_39 = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default_1, input_40);  addmm_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_2, add_tensor_1, alpha = 1);  view_default_2 = add_tensor_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_2, input_41);  add_tensor_2 = input_41 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_42, alpha = 1);  mul_tensor_1 = input_42 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_3, input_33)
        permute_default_1 = torch.ops.aten.permute.default(input_43, [1, 0]);  input_43 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_44, view_default_3, permute_default_1, beta = 1, alpha = 1);  input_44 = view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_2, input_45);  addmm_default_2 = None
        relu_default = torch.ops.aten.relu.default(view_default_4);  view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(relu_default, input_37);  relu_default = input_37 = None
        permute_default_2 = torch.ops.aten.permute.default(input_46, [1, 0]);  input_46 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_47, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_47 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_3, input_40);  addmm_default_3 = input_40 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_6, add_tensor_3, alpha = 1);  view_default_6 = add_tensor_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_4, input_48);  add_tensor_4 = input_48 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_2, input_49, alpha = 1);  mul_tensor_2 = input_49 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_5, input_33);  add_tensor_5 = input_33 = None
        permute_default_3 = torch.ops.aten.permute.default(input_50, [1, 0]);  input_50 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_51, view_default_7, permute_default_3, beta = 1, alpha = 1);  input_51 = view_default_7 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_4, input_45);  addmm_default_4 = input_45 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_8, input_52, alpha = 1);  view_default_8 = input_52 = None
        return (addmm_default, add_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
