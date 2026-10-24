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
    def forward(self, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50):
        mul_tensor = torch.ops.aten.mul.Tensor(input_32, input_33);  input_32 = input_33 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_34, -1);  input_34 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, unsqueeze_default, alpha = 1);  mul_tensor = unsqueeze_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        relu_default = torch.ops.aten.relu.default(convert_element_type_default);  convert_element_type_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_35, input_36, alpha = 1);  input_35 = input_36 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        relu_default_1 = torch.ops.aten.relu.default(convert_element_type_default_1);  convert_element_type_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(input_37, input_38, alpha = 1);  input_37 = input_38 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        relu_default_2 = torch.ops.aten.relu.default(convert_element_type_default_2);  convert_element_type_default_2 = None
        cat_default = torch.ops.aten.cat.default([relu_default_1, relu_default_2], 1);  relu_default_1 = relu_default_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_39, input_40, alpha = 1);  input_39 = input_40 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        relu_default_3 = torch.ops.aten.relu.default(convert_element_type_default_3);  convert_element_type_default_3 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(input_41, input_42, alpha = 1);  input_41 = input_42 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        relu_default_4 = torch.ops.aten.relu.default(convert_element_type_default_4);  convert_element_type_default_4 = None
        cat_default_1 = torch.ops.aten.cat.default([relu_default_3, relu_default_4], 1);  relu_default_3 = relu_default_4 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_43, input_44);  input_43 = input_44 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_45, -1);  input_45 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_1, unsqueeze_default_1, alpha = 1);  mul_tensor_1 = unsqueeze_default_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        relu_default_5 = torch.ops.aten.relu.default(convert_element_type_default_5);  convert_element_type_default_5 = None
        cat_default_2 = torch.ops.aten.cat.default([relu_default, cat_default, cat_default_1, relu_default_5], 1);  relu_default = cat_default = cat_default_1 = relu_default_5 = None
        mean_dim = torch.ops.aten.mean.dim(cat_default_2, [-1, -2], True, dtype = None);  cat_default_2 = None
        as_strided_default = torch.ops.aten.as_strided.default(mean_dim, input_46, input_47, None);  mean_dim = input_46 = input_47 = None
        view_default = torch.ops.aten.view.default(as_strided_default, input_48);  as_strided_default = input_48 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_49, torch.bfloat16);  input_49 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default_6 = view_default = permute_default = None
        return (addmm_default,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
