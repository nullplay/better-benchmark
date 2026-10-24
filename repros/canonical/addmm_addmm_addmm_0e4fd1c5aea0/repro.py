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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66):
        add_tensor = torch.ops.aten.add.Tensor(input_45, input_46, alpha = 1);  input_45 = input_46 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_47, torch.bfloat16);  input_47 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_48);  convert_element_type_default_1 = input_48 = None
        permute_default = torch.ops.aten.permute.default(input_49, [1, 0]);  input_49 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_50);  addmm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        permute_default_1 = torch.ops.aten.permute.default(input_52, [1, 0]);  input_52 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_2, view_default, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_2 = permute_default_1 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default_1, input_50);  addmm_default_1 = None
        convolution_default = torch.ops.aten.convolution.default(input_53, input_54, None, [1], [4], [1], False, [0], 768);  input_53 = input_54 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_55, torch.bfloat16);  input_55 = None
        convolution_default_1 = torch.ops.aten.convolution.default(convolution_default, convert_element_type_default_3, None, [1], [0], [1], False, [0], 1);  convolution_default = convert_element_type_default_3 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(convolution_default_1, input_56, alpha = 1);  convolution_default_1 = input_56 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_57, torch.bfloat16);  input_57 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_6, [1, 0]);  convert_element_type_default_6 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_5 = permute_default_2 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_2, input_50)
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_59);  input_59 = None
        permute_default_3 = torch.ops.aten.permute.default(convert_element_type_default_4, [0, 2, 1]);  convert_element_type_default_4 = None
        mul_tensor = torch.ops.aten.mul.Tensor(permute_default_3, view_default_3);  permute_default_3 = view_default_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_60, torch.bfloat16);  input_60 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_61, torch.bfloat16);  input_61 = None
        permute_default_4 = torch.ops.aten.permute.default(convert_element_type_default_8, [1, 0]);  convert_element_type_default_8 = None
        clone_default = torch.ops.aten.clone.default(mul_tensor, memory_format = torch.contiguous_format);  mul_tensor = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_62);  clone_default = input_62 = None
        mm_default = torch.ops.aten.mm.default(view_default_5, permute_default_4);  permute_default_4 = None
        view_default_6 = torch.ops.aten.view.default(mm_default, input_63);  input_63 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_6, convert_element_type_default_7, alpha = 1);  view_default_6 = convert_element_type_default_7 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_2, input_64);  add_tensor_2 = input_64 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32);  view_default_7 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_9, [1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_9, amax_default, alpha = 1);  convert_element_type_default_9 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(input_65, torch.bfloat16);  input_65 = None
        permute_default_5 = torch.ops.aten.permute.default(input_66, [1, 0]);  input_66 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_10, view_default, permute_default_5, beta = 1, alpha = 1);  convert_element_type_default_10 = view_default = permute_default_5 = None
        view_default_8 = torch.ops.aten.view.default(addmm_default_3, input_50);  addmm_default_3 = input_50 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        permute_default_6 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        return (view_default_1, view_default_2, addmm_default_2, view_default_5, mm_default, amax_default, sum_dim_int_list, view_default_8, convert_element_type_default_11, permute_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
