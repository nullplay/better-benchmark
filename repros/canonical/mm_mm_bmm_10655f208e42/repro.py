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
    def forward(self, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63):
        add_tensor = torch.ops.aten.add.Tensor(input_39, input_40, alpha = 1);  input_39 = input_40 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_41, input_42);  input_41 = input_42 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        div_scalar = torch.ops.aten.div.Scalar(input_43, 512);  input_43 = None
        mul_scalar = torch.ops.aten.mul.Scalar(input_44, 2.0);  input_44 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(div_scalar, mul_scalar);  div_scalar = mul_scalar = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_1, alpha = 1);  add_tensor_1 = mul_tensor_1 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(input_45, input_46, alpha = 1);  input_45 = input_46 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_3, input_47);  add_tensor_3 = input_47 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_48);  mul_tensor_2 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_3, alpha = 1);  add_tensor_2 = mul_tensor_3 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(input_48, 3);  input_48 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_49, pow_tensor_scalar);  input_49 = pow_tensor_scalar = None
        expand_default = torch.ops.aten.expand.default(mul_tensor_4, input_50, implicit = False);  mul_tensor_4 = input_50 = None
        div_scalar_1 = torch.ops.aten.div.Scalar(expand_default, 512);  expand_default = None
        pow_tensor_scalar_1 = torch.ops.aten.pow.Tensor_Scalar(input_51, 1.0);  input_51 = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(pow_tensor_scalar_1, 2.0);  pow_tensor_scalar_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(div_scalar_1, mul_scalar_1);  div_scalar_1 = mul_scalar_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, mul_tensor_5, alpha = 1);  add_tensor_4 = mul_tensor_5 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.bfloat16);  add_tensor_5 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_52, torch.bfloat16);  input_52 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_6);  convert_element_type_default = mul_tensor_6 = None
        view_default = torch.ops.aten.view.default(mul_tensor_7, input_53);  mul_tensor_7 = input_53 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_54);  permute_default = input_54 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_55);  view_default = input_55 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_56);  mm_default_1 = input_56 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_57);  view_default_1 = input_57 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_58);  clone_default = input_58 = None
        bmm_default = torch.ops.aten.bmm.default(input_59, view_default_3);  input_59 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_60);  view_default_3 = input_60 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_61);  bmm_default = input_61 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default_1, input_62);  bmm_default_1 = input_62 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(view_default_5, input_63);  view_default_5 = input_63 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        return (convert_element_type_default_2, mul_tensor_8, permute_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
