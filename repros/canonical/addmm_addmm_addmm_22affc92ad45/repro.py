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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_45, torch.bfloat16);  input_45 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_46, torch.bfloat16);  input_46 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_47, torch.bfloat16);  input_47 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_48);  convert_element_type_default_2 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_49)
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_1, input_50);  view_default_1 = input_50 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, input_51, alpha = 1);  mul_tensor = input_51 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_52, view_default, input_53, beta = 1, alpha = 1);  input_52 = input_53 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_54, view_default, input_55, beta = 1, alpha = 1);  input_54 = view_default = input_55 = None
        expand_default = torch.ops.aten.expand.default(input_56, input_57, implicit = False);  input_56 = input_57 = None
        view_default_2 = torch.ops.aten.view.default(expand_default, input_58);  expand_default = input_58 = None
        clone_default = torch.ops.aten.clone.default(input_59, memory_format = torch.contiguous_format);  input_59 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_60);  clone_default = input_60 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_2, view_default_3);  view_default_2 = view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_61);  bmm_default = input_61 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(clone_default_1, input_62);  clone_default_1 = input_62 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_63, torch.bfloat16);  input_63 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_64, torch.bfloat16);  input_64 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_65);  view_default_5 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0]);  convert_element_type_default_4 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_6, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(addmm_default_3, input_49)
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_7, add_tensor, alpha = 1);  view_default_7 = add_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_1, input_66);  add_tensor_1 = input_66 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_67, alpha = 1);  mul_tensor_1 = input_67 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_68, torch.bfloat16);  input_68 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16)
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_6, input_65);  convert_element_type_default_6 = input_65 = None
        permute_default_3 = torch.ops.aten.permute.default(input_69, [1, 0]);  input_69 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default_8, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_5 = permute_default_3 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default_4, input_70);  addmm_default_4 = input_70 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_71, torch.bfloat16);  input_71 = None
        view_default_10 = torch.ops.aten.view.default(input_72, input_48);  input_72 = input_48 = None
        permute_default_4 = torch.ops.aten.permute.default(input_73, [1, 0]);  input_73 = None
        addmm_default_5 = torch.ops.aten.addmm.default(convert_element_type_default_7, view_default_10, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_7 = view_default_10 = permute_default_4 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_5, input_49);  addmm_default_5 = input_49 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(view_default_11, add_tensor_2, alpha = 1);  view_default_11 = add_tensor_2 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_3, input_74);  input_74 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_2, input_75, alpha = 1);  mul_tensor_2 = input_75 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16)
        add_tensor_5 = torch.ops.aten.add.Tensor(input_76, add_tensor_4, alpha = 1);  input_76 = add_tensor_4 = None
        return (addmm_default, addmm_default_1, addmm_default_2, view_default_6, addmm_default_3, view_default_8, view_default_9, add_tensor_3, convert_element_type_default_8, add_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
