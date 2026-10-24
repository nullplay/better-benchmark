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
    def forward(self, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64):
        add_tensor = torch.ops.aten.add.Tensor(input_37, input_38, alpha = 1);  input_37 = input_38 = None
        expand_default = torch.ops.aten.expand.default(input_39, input_40, implicit = False);  input_39 = input_40 = None
        view_default = torch.ops.aten.view.default(expand_default, input_41);  expand_default = input_41 = None
        clone_default = torch.ops.aten.clone.default(input_42, memory_format = torch.contiguous_format);  input_42 = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_43);  clone_default = input_43 = None
        bmm_default = torch.ops.aten.bmm.default(view_default, view_default_1);  view_default = view_default_1 = None
        view_default_2 = torch.ops.aten.view.default(bmm_default, input_44);  bmm_default = input_44 = None
        permute_default = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_3 = torch.ops.aten.view.default(clone_default_1, input_45);  clone_default_1 = input_45 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_46);  view_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(input_47, [1, 0]);  input_47 = None
        addmm_default = torch.ops.aten.addmm.default(input_48, view_default_4, permute_default_1, beta = 1, alpha = 1);  input_48 = view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default, input_49);  addmm_default = None
        view_default_6 = torch.ops.aten.view.default(add_tensor, input_50);  add_tensor = None
        permute_default_2 = torch.ops.aten.permute.default(input_51, [1, 0]);  input_51 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_52, view_default_6, permute_default_2, beta = 1, alpha = 1);  input_52 = view_default_6 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_1, input_49);  addmm_default_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_7, input_53);  view_default_7 = input_53 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, input_54, alpha = 1);  mul_tensor = input_54 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_5, add_tensor_1, alpha = 1);  view_default_5 = add_tensor_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_2, input_55);  add_tensor_2 = input_55 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_56, alpha = 1);  mul_tensor_1 = input_56 = None
        view_default_8 = torch.ops.aten.view.default(add_tensor_3, input_46)
        permute_default_3 = torch.ops.aten.permute.default(input_57, [1, 0]);  input_57 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_58, view_default_8, permute_default_3, beta = 1, alpha = 1);  input_58 = view_default_8 = permute_default_3 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default_2, input_59);  addmm_default_2 = input_59 = None
        relu_default = torch.ops.aten.relu.default(view_default_9);  view_default_9 = None
        view_default_10 = torch.ops.aten.view.default(relu_default, input_50);  relu_default = input_50 = None
        permute_default_4 = torch.ops.aten.permute.default(input_60, [1, 0]);  input_60 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_61, view_default_10, permute_default_4, beta = 1, alpha = 1);  input_61 = view_default_10 = permute_default_4 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_3, input_49);  addmm_default_3 = input_49 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_11, add_tensor_3, alpha = 1);  view_default_11 = add_tensor_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_4, input_62);  add_tensor_4 = input_62 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_2, input_63, alpha = 1);  mul_tensor_2 = input_63 = None
        view_default_12 = torch.ops.aten.view.default(add_tensor_5, input_46);  input_46 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(input_64, add_tensor_5, alpha = 1);  input_64 = add_tensor_5 = None
        return (view_default_12, add_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
