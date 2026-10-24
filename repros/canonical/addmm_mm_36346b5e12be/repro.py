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
    def forward(self, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50):
        embedding_default = torch.ops.aten.embedding.default(input_38, input_39, 0, False, False);  input_39 = None
        slice_tensor = torch.ops.aten.slice.Tensor(embedding_default, 1, 1, 9223372036854775807, 1)
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(slice_tensor, [0, 0, 0, 1, 0, 0], 0.0);  slice_tensor = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(embedding_default, 1, 0, -1, 1)
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(slice_tensor_1, [0, 0, 1, 0, 0, 0], 0.0);  slice_tensor_1 = None
        cat_default = torch.ops.aten.cat.default([constant_pad_nd_default, embedding_default, constant_pad_nd_default_1], 2);  constant_pad_nd_default = embedding_default = constant_pad_nd_default_1 = None
        view_default = torch.ops.aten.view.default(cat_default, input_40);  cat_default = input_40 = None
        addmm_default = torch.ops.aten.addmm.default(input_41, view_default, input_42, beta = 1, alpha = 1);  input_41 = view_default = input_42 = None
        relu_default = torch.ops.aten.relu.default(input_43);  input_43 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(relu_default, torch.float32);  relu_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_44);  mul_tensor = input_44 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_45, alpha = 1);  mul_tensor_1 = input_45 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_46);  convert_element_type_default_1 = input_46 = None
        permute_default = torch.ops.aten.permute.default(input_38, [1, 0]);  input_38 = None
        cat_default_1 = torch.ops.aten.cat.default([permute_default, input_47], 0);  permute_default = input_47 = None
        constant_pad_nd_default_2 = torch.ops.aten.constant_pad_nd.default(cat_default_1, [0, 6, 0, 0], 0);  cat_default_1 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, constant_pad_nd_default_2);  view_default_1 = constant_pad_nd_default_2 = None
        slice_tensor_2 = torch.ops.aten.slice.Tensor(mm_default, 1, 0, -6, 1);  mm_default = None
        view_default_2 = torch.ops.aten.view.default(slice_tensor_2, input_48);  slice_tensor_2 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(view_default_2, input_49, alpha = 1);  view_default_2 = input_49 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_2, input_50);  add_tensor_2 = input_50 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32)
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_2, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default, alpha = 1);  convert_element_type_default_2 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, log_default, alpha = 1);  sub_tensor_1 = log_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(sub_tensor_2, torch.bfloat16);  sub_tensor_2 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_48);  view_default_3 = input_48 = None
        return (addmm_default, convert_element_type_default_3, view_default_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
