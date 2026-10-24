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
    def forward(self, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62):
        embedding_default = torch.ops.aten.embedding.default(input_44, input_45, -1, False, False);  input_45 = None
        add_tensor = torch.ops.aten.add.Tensor(input_46, 0, alpha = 1);  input_46 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(add_tensor, 0);  add_tensor = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_47, unsqueeze_default, -1, False, False);  input_47 = unsqueeze_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1, alpha = 1);  embedding_default = embedding_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(input_48, input_49, input_50, beta = 1, alpha = 1);  input_48 = input_49 = input_50 = None
        view_default = torch.ops.aten.view.default(addmm_default, input_51);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default, add_tensor_1, alpha = 1);  view_default = add_tensor_1 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_52, alpha = 1);  convert_element_type_default = input_52 = None
        view_default_1 = torch.ops.aten.view.default(input_53, input_51);  input_53 = input_51 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_1, alpha = 1);  add_tensor_2 = view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        add_tensor_4 = torch.ops.aten.add.Tensor(input_54, add_tensor_3, alpha = 1);  input_54 = add_tensor_3 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(input_55, input_56, alpha = 1);  input_55 = input_56 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32);  add_tensor_5 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction_1[0]
        getitem_1 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, getitem_1, alpha = 1);  convert_element_type_default_2 = getitem_1 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_6);  add_tensor_6 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_57);  mul_tensor = input_57 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(mul_tensor_1, input_58, alpha = 1);  mul_tensor_1 = input_58 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_7, torch.bfloat16);  add_tensor_7 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_3, input_59);  convert_element_type_default_3 = input_59 = None
        permute_default = torch.ops.aten.permute.default(input_44, [1, 0]);  input_44 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 7, 0, 0], 0);  permute_default = None
        mm_default = torch.ops.aten.mm.default(view_default_2, constant_pad_nd_default);  view_default_2 = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -7, 1);  mm_default = None
        view_default_3 = torch.ops.aten.view.default(slice_tensor, input_60);  slice_tensor = input_60 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32)
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_4, input_61);  convert_element_type_default_4 = input_61 = None
        amax_default = torch.ops.aten.amax.default(view_default_4, [1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(view_default_4, amax_default, alpha = 1);  view_default_4 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_2)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, log_default, alpha = 1);  sub_tensor_2 = log_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_62, 1);  input_62 = None
        gather_default = torch.ops.aten.gather.default(sub_tensor_3, 1, unsqueeze_default_1, sparse_grad = False);  sub_tensor_3 = unsqueeze_default_1 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        return (var_mean_correction, sub_tensor, convert_element_type_default_1, add_tensor_4, view_default_3, squeeze_dim)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
