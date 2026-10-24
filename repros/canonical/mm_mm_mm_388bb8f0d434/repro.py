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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58):
        add_tensor = torch.ops.aten.add.Tensor(input_45, input_46, alpha = 1);  input_45 = input_46 = None
        view_default = torch.ops.aten.view.default(input_47, input_48);  input_47 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default, alpha = 1);  add_tensor = view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_49);  mul_tensor = input_49 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_50, alpha = 1);  mul_tensor_1 = input_50 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_52);  convert_element_type_default_1 = input_52 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default);  permute_default = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_48);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_53, torch.bfloat16);  input_53 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_1, permute_default_1);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_48);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_54, torch.bfloat16);  input_54 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_1, permute_default_2);  view_default_1 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(mm_default_2, input_48);  mm_default_2 = input_48 = None
        view_default_5 = torch.ops.aten.view.default(view_default_2, input_55);  view_default_2 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(view_default_3, input_55);  view_default_3 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(view_default_4, input_55);  view_default_4 = input_55 = None
        permute_default_5 = torch.ops.aten.permute.default(view_default_7, [0, 2, 1, 3]);  view_default_7 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(permute_default_4, torch.float32);  permute_default_4 = None
        permute_default_6 = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 1, 3, 2]);  convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(permute_default_6, torch.bfloat16);  permute_default_6 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(permute_default_3, torch.bfloat16);  permute_default_3 = None
        expand_default = torch.ops.aten.expand.default(convert_element_type_default_6, input_56, implicit = False);  convert_element_type_default_6 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_8 = torch.ops.aten.view.default(input_57, input_58);  input_57 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_5, input_56, implicit = False);  permute_default_5 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_1, input_58);  clone_default_1 = input_58 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_8, view_default_9);  view_default_8 = None
        view_default_10 = torch.ops.aten.view.default(bmm_default, input_56);  bmm_default = input_56 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_9, [0, 2, 1]);  view_default_9 = None
        return (convert_element_type_default_5, clone_default, view_default_10, permute_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
