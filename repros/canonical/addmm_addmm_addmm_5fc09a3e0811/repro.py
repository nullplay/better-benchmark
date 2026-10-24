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
    def forward(self, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63):
        mul_tensor = torch.ops.aten.mul.Tensor(input_46, input_47);  input_46 = input_47 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        mean_dim = torch.ops.aten.mean.dim(mul_tensor_1, [-1], True, dtype = None)
        var_correction = torch.ops.aten.var.correction(mul_tensor_1, [-1], correction = 1.0, keepdim = True)
        sqrt_default = torch.ops.aten.sqrt.default(var_correction);  var_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, mean_dim, alpha = 1);  mul_tensor_1 = mean_dim = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(input_48, sub_tensor);  input_48 = sub_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(sqrt_default, 1e-06, alpha = 1);  sqrt_default = None
        div_tensor = torch.ops.aten.div.Tensor(mul_tensor_2, add_tensor);  mul_tensor_2 = add_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(div_tensor, input_49, alpha = 1);  div_tensor = input_49 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_52);  convert_element_type_default_2 = input_52 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_53);  addmm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_54);  view_default_1 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_55, torch.bfloat16);  input_55 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_56, torch.bfloat16);  input_56 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_53);  addmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_54);  view_default_3 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_57, torch.bfloat16);  input_57 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_6, [1, 0]);  convert_element_type_default_6 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_5 = view_default = permute_default_4 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_53);  addmm_default_2 = input_53 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_54);  view_default_5 = input_54 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        permute_default_6 = torch.ops.aten.permute.default(permute_default_3, [0, 1, 3, 2]);  permute_default_3 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_59, implicit = False);  permute_default_1 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_60);  clone_default = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_6, input_61, implicit = False);  permute_default_6 = input_61 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_8 = torch.ops.aten.view.default(clone_default_1, input_62);  clone_default_1 = input_62 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_7, view_default_8)
        view_default_9 = torch.ops.aten.view.default(bmm_default, input_63);  bmm_default = input_63 = None
        expand_default_2 = torch.ops.aten.expand.default(permute_default_5, input_59, implicit = False);  permute_default_5 = input_59 = None
        clone_default_2 = torch.ops.aten.clone.default(expand_default_2, memory_format = torch.contiguous_format);  expand_default_2 = None
        view_default_10 = torch.ops.aten.view.default(clone_default_2, input_60);  clone_default_2 = input_60 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1]);  view_default_7 = None
        return (view_default_8, view_default_9, view_default_10, permute_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
