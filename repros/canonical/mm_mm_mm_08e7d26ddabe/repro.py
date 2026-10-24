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
    def forward(self, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62):
        sub_tensor = torch.ops.aten.sub.Tensor(input_41, input_42, alpha = 1);  input_41 = input_42 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(input_43);  input_43 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_44);  mul_tensor = input_44 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, input_45, alpha = 1);  mul_tensor_1 = input_45 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0, 2])
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format)
        view_default = torch.ops.aten.view.default(clone_default, input_46);  clone_default = None
        permute_default_1 = torch.ops.aten.permute.default(input_47, [1, 0]);  input_47 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default_1);  view_default = permute_default_1 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_48);  mm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, input_49, alpha = 1);  view_default_1 = input_49 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format)
        view_default_2 = torch.ops.aten.view.default(clone_default_1, input_46);  clone_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(input_50, [1, 0]);  input_50 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, permute_default_2);  view_default_2 = permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_48);  mm_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_3, input_51, alpha = 1);  view_default_3 = input_51 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_4 = torch.ops.aten.view.default(clone_default_2, input_46);  clone_default_2 = input_46 = None
        permute_default_3 = torch.ops.aten.permute.default(input_52, [1, 0]);  input_52 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_4, permute_default_3);  view_default_4 = permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(mm_default_2, input_48);  mm_default_2 = input_48 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_5, input_53, alpha = 1);  view_default_5 = input_53 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor_3, input_54);  add_tensor_3 = input_54 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_6, [1, 0, 2, 3]);  view_default_6 = None
        permute_default_5 = torch.ops.aten.permute.default(permute_default_4, [0, 2, 1, 3]);  permute_default_4 = None
        view_default_7 = torch.ops.aten.view.default(permute_default_5, input_55);  permute_default_5 = input_55 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default_7, [0, 0, 256, 256], -1.0);  view_default_7 = None
        as_strided_default = torch.ops.aten.as_strided.default(constant_pad_nd_default, input_56, input_57, None);  constant_pad_nd_default = input_56 = input_57 = None
        mm_default_3 = torch.ops.aten.mm.default(input_58, input_59);  input_58 = input_59 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_3, input_60);  mm_default_3 = input_60 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_8, input_61, alpha = 1);  view_default_8 = input_61 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default, alpha = 1);  add_tensor_4 = convert_element_type_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32);  add_tensor_5 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_1, alpha = 1);  convert_element_type_default_1 = getitem_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_1, input_62);  sub_tensor_1 = input_62 = None
        return (add_tensor_1, add_tensor_2, as_strided_default, getitem, mul_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
