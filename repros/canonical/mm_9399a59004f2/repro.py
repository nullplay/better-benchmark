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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44):
        iota_default = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_tensor = torch.ops.aten.add.Tensor(input_35, input_36, alpha = 1);  input_35 = input_36 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_37, 1.1111111111111112);  input_37 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_38);  mul_tensor_1 = input_38 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_2, input_39, alpha = 1);  mul_tensor_2 = input_39 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_40, torch.bfloat16);  input_40 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_41);  convert_element_type_default_1 = input_41 = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default);  view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_42);  mm_default = input_42 = None
        ne_scalar = torch.ops.aten.ne.Scalar(input_43, 0);  input_43 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(ne_scalar, torch.int32);  ne_scalar = None
        iota_default_1 = torch.ops.prims.iota.default(1024, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(iota_default_1, convert_element_type_default_2);  iota_default_1 = convert_element_type_default_2 = None
        argmax_default = torch.ops.aten.argmax.default(mul_tensor_3, -1, False);  mul_tensor_3 = None
        index_tensor = torch.ops.aten.index.Tensor(view_default_1, [iota_default, argmax_default]);  view_default_1 = iota_default = argmax_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(index_tensor, torch.float32)
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_3, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, amax_default, alpha = 1);  convert_element_type_default_3 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, log_default, alpha = 1);  sub_tensor_1 = log_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(sub_tensor_2, torch.bfloat16);  sub_tensor_2 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        gather_default = torch.ops.aten.gather.default(convert_element_type_default_5, 1, input_44, sparse_grad = False);  convert_element_type_default_5 = input_44 = None
        return (index_tensor, gather_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
