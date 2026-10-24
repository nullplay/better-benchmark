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
    def forward(self, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94):
        add_tensor = torch.ops.aten.add.Tensor(input_58, input_59, alpha = 1);  input_58 = input_59 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, input_60);  add_tensor = input_60 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, input_61, alpha = 1);  mul_tensor = input_61 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_62, torch.bfloat16);  input_62 = None
        view_default = torch.ops.aten.view.default(input_63, input_64);  input_63 = None
        permute_default = torch.ops.aten.permute.default(input_65, [1, 0]);  input_65 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_66);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_1, add_tensor_1, alpha = 1);  view_default_1 = add_tensor_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_2, input_67);  input_67 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_68, alpha = 1);  mul_tensor_1 = input_68 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_69, torch.bfloat16);  input_69 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_70, torch.bfloat16);  input_70 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16)
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_3, input_71);  convert_element_type_default_3 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_1, view_default_2, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_1 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_72)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_3, input_73);  view_default_3 = input_73 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_74, torch.bfloat16);  input_74 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_75, torch.bfloat16);  input_75 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_5, [1, 0]);  convert_element_type_default_5 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_2, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_2, input_72)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(view_default_4, input_76);  view_default_4 = input_76 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_77, torch.bfloat16);  input_77 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_78, torch.bfloat16);  input_78 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_2, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_3, input_72);  addmm_default_3 = None
        view_default_6 = torch.ops.aten.view.default(view_default_5, input_79);  view_default_5 = input_79 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(input_80, input_81, alpha = 1);  input_80 = input_81 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(add_tensor_4, input_82);  add_tensor_4 = input_82 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(mul_tensor_4, input_83, alpha = 1);  mul_tensor_4 = input_83 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_84, torch.bfloat16);  input_84 = None
        view_default_7 = torch.ops.aten.view.default(input_85, input_71);  input_85 = None
        permute_default_4 = torch.ops.aten.permute.default(input_86, [1, 0]);  input_86 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_8, view_default_7, permute_default_4, beta = 1, alpha = 1);  convert_element_type_default_8 = view_default_7 = permute_default_4 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_4, input_72);  addmm_default_4 = input_72 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(view_default_8, add_tensor_5, alpha = 1);  view_default_8 = add_tensor_5 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_6, input_87);  input_87 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_5, input_88, alpha = 1);  mul_tensor_5 = input_88 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_89, torch.bfloat16);  input_89 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_90, torch.bfloat16);  input_90 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        view_default_9 = torch.ops.aten.view.default(convert_element_type_default_11, input_64);  convert_element_type_default_11 = input_64 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0]);  convert_element_type_default_10 = None
        addmm_default_5 = torch.ops.aten.addmm.default(convert_element_type_default_9, view_default_9, permute_default_5, beta = 1, alpha = 1);  convert_element_type_default_9 = permute_default_5 = None
        view_default_10 = torch.ops.aten.view.default(addmm_default_5, input_66);  addmm_default_5 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_10, add_tensor_3, alpha = 1);  view_default_10 = add_tensor_3 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor_8, input_91);  input_91 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(mul_tensor_6, input_92, alpha = 1);  mul_tensor_6 = input_92 = None
        convert_element_type_default_12 = torch.ops.prims.convert_element_type.default(input_93, torch.bfloat16);  input_93 = None
        convert_element_type_default_13 = torch.ops.prims.convert_element_type.default(add_tensor_9, torch.bfloat16);  add_tensor_9 = None
        view_default_11 = torch.ops.aten.view.default(convert_element_type_default_13, input_71);  convert_element_type_default_13 = input_71 = None
        permute_default_6 = torch.ops.aten.permute.default(input_94, [1, 0]);  input_94 = None
        addmm_default_6 = torch.ops.aten.addmm.default(convert_element_type_default_12, view_default_11, permute_default_6, beta = 1, alpha = 1);  convert_element_type_default_12 = permute_default_6 = None
        view_default_12 = torch.ops.aten.view.default(addmm_default_6, input_66);  input_66 = None
        return (add_tensor_2, view_default_2, addmm_default_1, mul_tensor_2, addmm_default_2, mul_tensor_3, view_default_6, add_tensor_6, view_default_9, add_tensor_8, view_default_11, addmm_default_6, view_default_12)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
