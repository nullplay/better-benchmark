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
    def forward(self, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70):
        add_tensor = torch.ops.aten.add.Tensor(input_48, input_49, alpha = 1);  input_48 = input_49 = None
        view_default = torch.ops.aten.view.default(input_50, input_51);  input_50 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        addmm_default = torch.ops.aten.addmm.default(input_52, input_53, input_54, beta = 1, alpha = 1);  input_52 = input_53 = input_54 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_51);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_1, alpha = 1);  add_tensor_1 = view_default_1 = None
        mean_dim = torch.ops.aten.mean.dim(add_tensor_2, [-1], True, dtype = None)
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, mean_dim, alpha = 1);  mean_dim = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_55, sub_tensor);  input_55 = sub_tensor = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_correction = torch.ops.aten.var.correction(convert_element_type_default, [-1], correction = 1.0, keepdim = True);  convert_element_type_default = None
        sqrt_default = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(sqrt_default, torch.bfloat16);  sqrt_default = None
        add_tensor_3 = torch.ops.aten.add.Tensor(convert_element_type_default_1, 1e-06, alpha = 1);  convert_element_type_default_1 = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor, add_tensor_3);  mul_tensor = add_tensor_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(div_tensor, input_56, alpha = 1);  div_tensor = input_56 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_4, input_57)
        permute_default = torch.ops.aten.permute.default(input_58, [1, 0]);  input_58 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_59, view_default_2, permute_default, beta = 1, alpha = 1);  input_59 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_51);  addmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_60);  view_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_61, implicit = False);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_4, input_57)
        permute_default_2 = torch.ops.aten.permute.default(input_62, [1, 0]);  input_62 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_63, view_default_5, permute_default_2, beta = 1, alpha = 1);  input_63 = view_default_5 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_51);  addmm_default_2 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_60);  view_default_6 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        permute_default_4 = torch.ops.aten.permute.default(permute_default_3, [0, 1, 3, 2]);  permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(input_64, input_65);  input_64 = input_65 = None
        view_default_9 = torch.ops.aten.view.default(add_tensor_4, input_57);  add_tensor_4 = input_57 = None
        permute_default_5 = torch.ops.aten.permute.default(input_66, [1, 0]);  input_66 = None
        addmm_default_3 = torch.ops.aten.addmm.default(input_67, view_default_9, permute_default_5, beta = 1, alpha = 1);  input_67 = view_default_9 = permute_default_5 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_3, input_51);  addmm_default_3 = None
        view_default_11 = torch.ops.aten.view.default(view_default_10, input_60);  view_default_10 = input_60 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_11, [0, 2, 1, 3]);  view_default_11 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_6, input_61, implicit = False);  permute_default_6 = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_12 = torch.ops.aten.view.default(clone_default, input_68);  clone_default = input_68 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_8, view_default_12);  view_default_8 = view_default_12 = None
        view_default_13 = torch.ops.aten.view.default(bmm_default, input_61);  bmm_default = input_61 = None
        view_default_14 = torch.ops.aten.view.default(input_69, input_51);  input_69 = input_51 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_14, alpha = 1);  add_tensor_2 = view_default_14 = None
        mean_dim_1 = torch.ops.aten.mean.dim(add_tensor_5, [-1], True, dtype = None)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_5, mean_dim_1, alpha = 1);  mean_dim_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, input_70, alpha = 1);  add_tensor_5 = input_70 = None
        return (expand_default, permute_default_4, view_default_13, sub_tensor_1, convert_element_type_default_2, add_tensor_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
