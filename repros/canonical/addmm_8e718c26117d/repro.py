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
    def forward(self, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40):
        add_tensor = torch.ops.aten.add.Tensor(input_29, input_30, alpha = 1);  input_29 = input_30 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_31);  mul_tensor = input_31 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_32, alpha = 1);  mul_tensor_1 = input_32 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_33);  convert_element_type_default_1 = input_33 = None
        permute_default = torch.ops.aten.permute.default(input_34, [1, 0]);  input_34 = None
        addmm_default = torch.ops.aten.addmm.default(input_35, view_default, permute_default, beta = 1, alpha = 1);  input_35 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_36);  addmm_default = input_36 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32)
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_2, input_37);  convert_element_type_default_2 = input_37 = None
        amax_default = torch.ops.aten.amax.default(view_default_2, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(view_default_2, amax_default, alpha = 1);  view_default_2 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, log_default, alpha = 1);  sub_tensor_1 = log_default = None
        where_self = torch.ops.aten.where.self(input_38, input_39, input_40);  input_38 = input_39 = input_40 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default = torch.ops.aten.gather.default(sub_tensor_2, 1, unsqueeze_default, sparse_grad = False);  sub_tensor_2 = unsqueeze_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        return (view_default_1, neg_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
