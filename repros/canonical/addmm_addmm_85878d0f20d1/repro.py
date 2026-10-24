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
    def forward(self, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59):
        fmod_scalar = torch.ops.aten.fmod.Scalar(input_40, 14);  input_40 = None
        view_default = torch.ops.aten.view.default(input_41, input_42);  input_41 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_43, input_44);  input_43 = input_44 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, mul_tensor, alpha = 1);  view_default = mul_tensor = None
        view_default_1 = torch.ops.aten.view.default(add_tensor, input_45);  add_tensor = None
        index_tensor = torch.ops.aten.index.Tensor(input_46, [None, fmod_scalar]);  input_46 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(index_tensor, [None, None, fmod_scalar]);  index_tensor = fmod_scalar = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_47, torch.bfloat16);  input_47 = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default, 0.9086956530809402);  convert_element_type_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(index_tensor_1, div_tensor);  index_tensor_1 = div_tensor = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, mul_tensor_1, alpha = 1);  view_default_1 = mul_tensor_1 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_1, input_42);  add_tensor_1 = input_42 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_48, torch.bfloat16);  input_48 = None
        view_default_3 = torch.ops.aten.view.default(input_49, input_50);  input_49 = input_50 = None
        permute_default = torch.ops.aten.permute.default(input_51, [1, 0]);  input_51 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_3, permute_default, beta = 1, alpha = 1);  convert_element_type_default_1 = view_default_3 = permute_default = None
        view_default_4 = torch.ops.aten.view.default(addmm_default, input_52);  input_52 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_4, torch.float32);  view_default_4 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.5)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, 0.7071067811865476);  convert_element_type_default_2 = None
        erf_default = torch.ops.aten.erf.default(mul_tensor_3);  mul_tensor_3 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(erf_default, 1, alpha = 1);  erf_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_2, add_tensor_2);  mul_tensor_2 = add_tensor_2 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_53, torch.bfloat16);  input_53 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_54, torch.bfloat16);  input_54 = None
        view_default_5 = torch.ops.aten.view.default(convert_element_type_default_3, input_55);  convert_element_type_default_3 = input_55 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_5, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_1 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_56);  addmm_default_1 = input_56 = None
        lt_scalar = torch.ops.aten.lt.Scalar(input_57, 0.9086956530809402);  input_57 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(lt_scalar, torch.bfloat16);  lt_scalar = None
        div_tensor_1 = torch.ops.aten.div.Tensor(convert_element_type_default_6, 0.9086956530809402);  convert_element_type_default_6 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(view_default_6, div_tensor_1);  view_default_6 = div_tensor_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_2, mul_tensor_5, alpha = 1);  view_default_2 = mul_tensor_5 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_3, input_45);  add_tensor_3 = input_45 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_58);  view_default_7 = input_58 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_8, [0, 1, 3, 4, 2, 5]);  view_default_8 = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_9 = torch.ops.aten.view.default(clone_default, input_59);  clone_default = input_59 = None
        return (addmm_default, view_default_5, view_default_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
