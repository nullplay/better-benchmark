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
    def forward(self, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_32, torch.bfloat16);  input_32 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, getitem_1, alpha = 1);  convert_element_type_default_1 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_33);  mul_tensor = input_33 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_34, alpha = 1);  mul_tensor_1 = input_34 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_35, torch.bfloat16);  input_35 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_36, torch.bfloat16);  input_36 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_4, input_37);  convert_element_type_default_4 = input_37 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_3, [1, 0]);  convert_element_type_default_3 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 6, 0, 0], 0);  permute_default = None
        full_default = torch.ops.aten.full.default(input_38, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_38 = None
        cat_default = torch.ops.aten.cat.default([convert_element_type_default_2, full_default], 0);  convert_element_type_default_2 = full_default = None
        addmm_default = torch.ops.aten.addmm.default(cat_default, view_default, constant_pad_nd_default, beta = 1, alpha = 1);  cat_default = view_default = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -6, 1);  addmm_default = None
        view_default_1 = torch.ops.aten.view.default(slice_tensor, input_39);  slice_tensor = input_39 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_40);  input_40 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_5, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, amax_default, alpha = 1);  convert_element_type_default_5 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, log_default, alpha = 1);  sub_tensor_1 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(sub_tensor_2, torch.bfloat16);  sub_tensor_2 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(convert_element_type_default_6, torch.float32);  convert_element_type_default_6 = None
        return (view_default_1, amax_default, log_default, convert_element_type_default_7)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
