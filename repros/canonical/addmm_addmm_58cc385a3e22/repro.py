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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67):
        embedding_default = torch.ops.aten.embedding.default(input_45, input_46, 0, False, False);  input_46 = None
        full_default = torch.ops.aten.full.default(input_47, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_47 = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_48, full_default, -1, False, False);  input_48 = full_default = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1, alpha = 1);  embedding_default = embedding_default_1 = None
        embedding_default_2 = torch.ops.aten.embedding.default(input_49, input_50, -1, False, False);  input_49 = input_50 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2, alpha = 1);  add_tensor = embedding_default_2 = None
        addmm_default = torch.ops.aten.addmm.default(input_51, input_52, input_53, beta = 1, alpha = 1);  input_51 = input_52 = input_53 = None
        view_default = torch.ops.aten.view.default(addmm_default, input_54);  addmm_default = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default, alpha = 1);  add_tensor_1 = view_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_55, alpha = 1);  convert_element_type_default = input_55 = None
        view_default_1 = torch.ops.aten.view.default(input_56, input_54);  input_56 = input_54 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, view_default_1, alpha = 1);  add_tensor_2 = view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.float32)
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, input_57, alpha = 1);  add_tensor_3 = input_57 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_58, torch.bfloat16);  input_58 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.float32);  convert_element_type_default_2 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction_1[0]
        getitem_1 = var_mean_correction_1[1];  var_mean_correction_1 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_1, alpha = 1);  convert_element_type_default_3 = getitem_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default);  sub_tensor_1 = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_59);  mul_tensor = input_59 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_1, input_60, alpha = 1);  mul_tensor_1 = input_60 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.bfloat16);  add_tensor_6 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_4, input_61);  convert_element_type_default_4 = input_61 = None
        permute_default = torch.ops.aten.permute.default(input_45, [1, 0]);  input_45 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_62, view_default_2, permute_default, beta = 1, alpha = 1);  input_62 = view_default_2 = permute_default = None
        view_default_3 = torch.ops.aten.view.default(addmm_default_1, input_63);  addmm_default_1 = input_63 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32)
        view_default_4 = torch.ops.aten.view.default(convert_element_type_default_5, input_64);  convert_element_type_default_5 = input_64 = None
        amax_default = torch.ops.aten.amax.default(view_default_4, [1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(view_default_4, amax_default, alpha = 1);  view_default_4 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_2)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, log_default, alpha = 1);  sub_tensor_2 = log_default = None
        where_self = torch.ops.aten.where.self(input_65, input_66, input_67);  input_65 = input_66 = input_67 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(where_self, 1);  where_self = None
        gather_default = torch.ops.aten.gather.default(sub_tensor_3, 1, unsqueeze_default, sparse_grad = False);  sub_tensor_3 = unsqueeze_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        neg_default = torch.ops.aten.neg.default(squeeze_dim);  squeeze_dim = None
        return (var_mean_correction, sub_tensor, convert_element_type_default_1, add_tensor_4, view_default_3, neg_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
