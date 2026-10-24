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
    def forward(self, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61):
        embedding_default = torch.ops.aten.embedding.default(input_41, input_42, 50257, False, False);  input_42 = None
        full_default = torch.ops.aten.full.default(input_43, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_43 = None
        index_tensor = torch.ops.aten.index.Tensor(input_44, [full_default]);  input_44 = full_default = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, index_tensor, alpha = 1);  embedding_default = index_tensor = None
        view_default = torch.ops.aten.view.default(input_45, input_46);  input_45 = None
        permute_default = torch.ops.aten.permute.default(input_47, [1, 0]);  input_47 = None
        addmm_default = torch.ops.aten.addmm.default(input_48, view_default, permute_default, beta = 1, alpha = 1);  input_48 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_49);  addmm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_1, alpha = 1);  add_tensor = view_default_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, input_50);  sub_tensor = input_50 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_51, input_52, input_53, beta = 1, alpha = 1);  input_51 = input_52 = input_53 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default_1, input_49);  addmm_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_2, alpha = 1);  add_tensor_1 = view_default_2 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, input_54, alpha = 1);  convert_element_type_default_1 = input_54 = None
        view_default_3 = torch.ops.aten.view.default(input_55, input_49);  input_55 = input_49 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_3, alpha = 1);  add_tensor_2 = view_default_3 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, input_56, alpha = 1);  add_tensor_3 = input_56 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(input_57, input_58, alpha = 1);  input_57 = input_58 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32);  add_tensor_5 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_2[0]
        getitem_3 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_3, alpha = 1);  convert_element_type_default_3 = getitem_3 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_59);  mul_tensor_1 = input_59 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_2, input_60, alpha = 1);  mul_tensor_2 = input_60 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_4, input_46);  convert_element_type_default_4 = input_46 = None
        permute_default_1 = torch.ops.aten.permute.default(input_41, [1, 0]);  input_41 = None
        mm_default = torch.ops.aten.mm.default(view_default_4, permute_default_1);  view_default_4 = permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(mm_default, input_61);  mm_default = input_61 = None
        return (getitem, mul_tensor, var_mean_correction_1, sub_tensor_1, convert_element_type_default_2, add_tensor_4, view_default_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
