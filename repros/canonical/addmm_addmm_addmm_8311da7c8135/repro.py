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
    def forward(self, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81):
        mul_tensor = torch.ops.aten.mul.Tensor(input_55, input_56);  input_55 = input_56 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, input_57, alpha = 1);  mul_tensor = input_57 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_59, torch.bfloat16);  input_59 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_60);  convert_element_type_default_2 = input_60 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_61);  addmm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_62, torch.bfloat16);  input_62 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_63)
        permute_default_1 = torch.ops.aten.permute.default(input_64, [1, 0]);  input_64 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_2, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_61);  addmm_default_1 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_65);  view_default_3 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_66, torch.bfloat16);  input_66 = None
        permute_default_3 = torch.ops.aten.permute.default(input_67, [1, 0]);  input_67 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_2, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_2, input_61);  addmm_default_2 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_65);  view_default_5 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_6, [0, 2, 1, 3]);  view_default_6 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_68, torch.bfloat16);  input_68 = None
        permute_default_5 = torch.ops.aten.permute.default(input_69, [1, 0]);  input_69 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default_2, permute_default_5, beta = 1, alpha = 1);  convert_element_type_default_5 = permute_default_5 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_3, input_61);  addmm_default_3 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_65);  view_default_7 = input_65 = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        expand_default = torch.ops.aten.expand.default(input_70, input_71, implicit = False);  input_70 = input_71 = None
        view_default_9 = torch.ops.aten.view.default(expand_default, input_72);  expand_default = input_72 = None
        clone_default = torch.ops.aten.clone.default(input_73, memory_format = torch.contiguous_format);  input_73 = None
        view_default_10 = torch.ops.aten.view.default(clone_default, input_74);  clone_default = input_74 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_9, view_default_10);  view_default_9 = view_default_10 = None
        view_default_11 = torch.ops.aten.view.default(bmm_default, input_75);  bmm_default = input_75 = None
        permute_default_7 = torch.ops.aten.permute.default(view_default_11, [0, 2, 1, 3]);  view_default_11 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_7, memory_format = torch.contiguous_format);  permute_default_7 = None
        view_default_12 = torch.ops.aten.view.default(clone_default_1, input_76);  clone_default_1 = input_76 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_77, torch.bfloat16);  input_77 = None
        view_default_13 = torch.ops.aten.view.default(view_default_12, input_63);  view_default_12 = input_63 = None
        permute_default_8 = torch.ops.aten.permute.default(input_78, [1, 0]);  input_78 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_13, permute_default_8, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_8 = None
        view_default_14 = torch.ops.aten.view.default(addmm_default_4, input_61);  addmm_default_4 = input_61 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_1, view_default_14, alpha = 1);  view_default_1 = view_default_14 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_7, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_7, getitem_1, alpha = 1);  convert_element_type_default_7 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_79);  mul_tensor_1 = input_79 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_2, input_80, alpha = 1);  mul_tensor_2 = input_80 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16)
        add_tensor_4 = torch.ops.aten.add.Tensor(input_81, add_tensor_3, alpha = 1);  input_81 = add_tensor_3 = None
        return (view_default_2, permute_default_2, permute_default_4, permute_default_6, view_default_13, add_tensor_1, getitem_1, rsqrt_default, convert_element_type_default_8, add_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
