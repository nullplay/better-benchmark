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
    def forward(self, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48):
        view_default = torch.ops.aten.view.default(input_34, input_35);  input_34 = input_35 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default, 0.5);  view_default = None
        tanh_default = torch.ops.aten.tanh.default(input_36);  input_36 = None
        add_tensor = torch.ops.aten.add.Tensor(tanh_default, 1.0, alpha = 1);  tanh_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, add_tensor);  mul_tensor = add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(mul_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        view_default_1 = torch.ops.aten.view.default(input_37, input_38);  input_37 = input_38 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, getitem_1, alpha = 1);  mul_tensor_1 = getitem_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_39);  mul_tensor_2 = input_39 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_3, input_40, alpha = 1);  mul_tensor_3 = input_40 = None
        view_default_2 = torch.ops.aten.view.default(add_tensor_2, input_41);  add_tensor_2 = input_41 = None
        permute_default = torch.ops.aten.permute.default(input_42, [1, 0]);  input_42 = None
        addmm_default = torch.ops.aten.addmm.default(input_43, view_default_2, permute_default, beta = 1, alpha = 1);  input_43 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default, input_44);  addmm_default = input_44 = None
        view_default_4 = torch.ops.aten.view.default(view_default_3, input_45);  input_45 = None
        amax_default = torch.ops.aten.amax.default(view_default_4, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(view_default_4, amax_default, alpha = 1);  view_default_4 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, log_default, alpha = 1);  sub_tensor_1 = log_default = None
        ne_scalar = torch.ops.aten.ne.Scalar(view_default_1, -100)
        full_default = torch.ops.aten.full.default(input_46, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_46 = None
        where_self = torch.ops.aten.where.self(ne_scalar, view_default_1, full_default);  ne_scalar = view_default_1 = full_default = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default = torch.ops.aten.gather.default(sub_tensor_2, 1, unsqueeze_default, sparse_grad = False);  sub_tensor_2 = unsqueeze_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        where_self_1 = torch.ops.aten.where.self(input_47, neg_default, input_48);  input_47 = neg_default = input_48 = None
        return (view_default_3, where_self_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
