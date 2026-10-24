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
    def forward(self, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65):
        view_default = torch.ops.aten.view.default(input_38, input_39);  input_38 = None
        add_tensor = torch.ops.aten.add.Tensor(input_40, input_41, alpha = 1);  input_40 = input_41 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default, add_tensor, alpha = 1);  view_default = add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_1, input_42);  add_tensor_1 = input_42 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor, input_43, alpha = 1);  mul_tensor = input_43 = None
        view_default_1 = torch.ops.aten.view.default(add_tensor_2, input_44)
        permute_default = torch.ops.aten.permute.default(input_45, [1, 0]);  input_45 = None
        addmm_default = torch.ops.aten.addmm.default(input_46, view_default_1, permute_default, beta = 1, alpha = 1);  input_46 = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_47);  addmm_default = None
        relu_default = torch.ops.aten.relu.default(view_default_2);  view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(relu_default, input_48);  relu_default = None
        permute_default_1 = torch.ops.aten.permute.default(input_49, [1, 0]);  input_49 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_50, view_default_3, permute_default_1, beta = 1, alpha = 1);  input_50 = view_default_3 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_39);  addmm_default_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_4, add_tensor_2, alpha = 1);  view_default_4 = add_tensor_2 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_3, input_51);  add_tensor_3 = input_51 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_1, input_52, alpha = 1);  mul_tensor_1 = input_52 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_4, input_44)
        permute_default_2 = torch.ops.aten.permute.default(input_53, [1, 0]);  input_53 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_54, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_54 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_47);  addmm_default_2 = None
        relu_default_1 = torch.ops.aten.relu.default(view_default_6);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(relu_default_1, input_48);  relu_default_1 = input_48 = None
        permute_default_3 = torch.ops.aten.permute.default(input_55, [1, 0]);  input_55 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_56, view_default_7, permute_default_3, beta = 1, alpha = 1);  input_56 = view_default_7 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_39);  addmm_default_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_8, add_tensor_4, alpha = 1);  view_default_8 = add_tensor_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_5, input_57);  add_tensor_5 = input_57 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_2, input_58, alpha = 1);  mul_tensor_2 = input_58 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_6, input_44);  input_44 = None
        permute_default_4 = torch.ops.aten.permute.default(input_59, [1, 0]);  input_59 = None
        addmm_default_4 = torch.ops.aten.addmm.default(input_60, view_default_9, permute_default_4, beta = 1, alpha = 1);  input_60 = view_default_9 = permute_default_4 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_4, input_47);  addmm_default_4 = input_47 = None
        addmm_default_5 = torch.ops.aten.addmm.default(input_61, input_62, input_63, beta = 1, alpha = 1);  input_61 = input_62 = input_63 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_5, input_39);  addmm_default_5 = input_39 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(view_default_11, add_tensor_6, alpha = 1);  view_default_11 = add_tensor_6 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_7, input_64);  add_tensor_7 = input_64 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(mul_tensor_3, input_65, alpha = 1);  mul_tensor_3 = input_65 = None
        return (view_default_10, add_tensor_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
