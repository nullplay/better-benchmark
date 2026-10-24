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
    def forward(self, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_49, torch.bfloat16);  input_49 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_50, torch.bfloat16);  input_50 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_52);  convert_element_type_default_2 = input_52 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default, beta = 1, alpha = 1);  convert_element_type_default = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_53);  addmm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_54, torch.bfloat16);  input_54 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_55)
        permute_default_1 = torch.ops.aten.permute.default(input_56, [1, 0]);  input_56 = None
        addmm_default_1 = torch.ops.aten.addmm.default(convert_element_type_default_3, view_default_2, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default_3 = permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_53);  addmm_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_57, torch.bfloat16);  input_57 = None
        permute_default_2 = torch.ops.aten.permute.default(input_58, [1, 0]);  input_58 = None
        addmm_default_2 = torch.ops.aten.addmm.default(convert_element_type_default_4, view_default_2, permute_default_2, beta = 1, alpha = 1);  convert_element_type_default_4 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default_2, input_53);  addmm_default_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_59, torch.bfloat16);  input_59 = None
        permute_default_3 = torch.ops.aten.permute.default(input_60, [1, 0]);  input_60 = None
        addmm_default_3 = torch.ops.aten.addmm.default(convert_element_type_default_5, view_default_2, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_5 = permute_default_3 = None
        view_default_5 = torch.ops.aten.view.default(addmm_default_3, input_53);  addmm_default_3 = None
        expand_default = torch.ops.aten.expand.default(input_61, input_62, implicit = False);  input_61 = input_62 = None
        view_default_6 = torch.ops.aten.view.default(expand_default, input_63);  expand_default = input_63 = None
        clone_default = torch.ops.aten.clone.default(input_64, memory_format = torch.contiguous_format);  input_64 = None
        view_default_7 = torch.ops.aten.view.default(clone_default, input_65);  clone_default = input_65 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_6, view_default_7);  view_default_6 = view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(bmm_default, input_66);  bmm_default = input_66 = None
        permute_default_4 = torch.ops.aten.permute.default(view_default_8, [0, 2, 1, 3]);  view_default_8 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_4, memory_format = torch.contiguous_format);  permute_default_4 = None
        view_default_9 = torch.ops.aten.view.default(clone_default_1, input_67);  clone_default_1 = input_67 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_68, torch.bfloat16);  input_68 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(input_69, torch.bfloat16);  input_69 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_55);  view_default_9 = input_55 = None
        permute_default_5 = torch.ops.aten.permute.default(convert_element_type_default_7, [1, 0]);  convert_element_type_default_7 = None
        addmm_default_4 = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_10, permute_default_5, beta = 1, alpha = 1);  convert_element_type_default_6 = permute_default_5 = None
        view_default_11 = torch.ops.aten.view.default(addmm_default_4, input_53);  addmm_default_4 = input_53 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(input_70, torch.bfloat16);  input_70 = None
        gt_scalar = torch.ops.aten.gt.Scalar(convert_element_type_default_8, 0.1);  convert_element_type_default_8 = None
        mul_tensor = torch.ops.aten.mul.Tensor(gt_scalar, view_default_11);  gt_scalar = view_default_11 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1.1111111111111112);  mul_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, view_default_1, alpha = 1);  mul_tensor_1 = view_default_1 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_9, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_9, getitem_1, alpha = 1);  convert_element_type_default_9 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_71);  mul_tensor_2 = input_71 = None
        return (view_default_2, view_default_3, view_default_4, view_default_5, view_default_10, add_tensor, getitem_1, rsqrt_default, mul_tensor_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
