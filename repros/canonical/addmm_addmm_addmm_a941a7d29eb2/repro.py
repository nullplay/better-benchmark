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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70):
        view_default = torch.ops.aten.view.default(input_45, input_46);  input_45 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default, input_47);  view_default = input_47 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, input_48, alpha = 1);  mul_tensor = input_48 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_49, torch.bfloat16);  input_49 = None
        view_default_1 = torch.ops.aten.view.default(input_50, input_51);  input_50 = None
        permute_default = torch.ops.aten.permute.default(input_52, [1, 0]);  input_52 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default_1, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_46)
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_2, add_tensor, alpha = 1);  view_default_2 = add_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_1, input_53);  add_tensor_1 = input_53 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_54, alpha = 1);  mul_tensor_1 = input_54 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_55, torch.bfloat16);  input_55 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_56, torch.bfloat16);  input_56 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16)
        view_default_3 = torch.ops.aten.view.default(convert_element_type_default_3, input_51);  convert_element_type_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_3, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_1 = permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_1, input_57);  addmm_default_1 = None
        relu_default = torch.ops.aten.relu.default(view_default_4);  view_default_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_59, torch.bfloat16);  input_59 = None
        view_default_5 = torch.ops.aten.view.default(relu_default, input_60)
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_5, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_2, input_46);  addmm_default_2 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_6, add_tensor_2, alpha = 1);  view_default_6 = add_tensor_2 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_3, input_61);  input_61 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_2, input_62, alpha = 1);  mul_tensor_2 = input_62 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_63, torch.bfloat16);  input_63 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16)
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_7, input_51);  convert_element_type_default_7 = input_51 = None
        permute_default_3 = torch.ops.aten.permute.default(input_64, [1, 0]);  input_64 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_7, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_3 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_57);  addmm_default_3 = input_57 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_65, torch.bfloat16);  input_65 = None
        view_default_9 = torch.ops.aten.view.default(input_66, input_60);  input_66 = input_60 = None
        permute_default_4 = torch.ops.aten.permute.default(input_67, [1, 0]);  input_67 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_8, view_default_9, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_8 = view_default_9 = permute_default_4 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_4, input_46);  addmm_default_4 = input_46 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(view_default_10, add_tensor_4, alpha = 1);  view_default_10 = add_tensor_4 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_5, input_68);  input_68 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_3, input_69, alpha = 1);  mul_tensor_3 = input_69 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16)
        add_tensor_7 = torch.ops.aten.add.Tensor(input_70, add_tensor_6, alpha = 1);  input_70 = add_tensor_6 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        return (addmm_default, view_default_3, view_default_5, add_tensor_3, view_default_7, view_default_8, add_tensor_5, convert_element_type_default_9, add_tensor_7, le_scalar)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
