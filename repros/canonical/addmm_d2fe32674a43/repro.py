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
    def forward(self, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42):
        add_tensor = torch.ops.aten.add.Tensor(input_33, input_34, alpha = 1);  input_33 = input_34 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.float32);  add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_1);  add_tensor_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_35);  mul_tensor = input_35 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(mul_tensor_1, input_36, alpha = 1);  mul_tensor_1 = input_36 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.bfloat16);  add_tensor_2 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_1, input_37);  convert_element_type_default_1 = input_37 = None
        permute_default = torch.ops.aten.permute.default(input_38, [1, 0]);  input_38 = None
        addmm_default = torch.ops.aten.addmm.default(input_39, view_default, permute_default, beta = 1, alpha = 1);  input_39 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_40);  addmm_default = input_40 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default_1, 1, -1);  view_default_1 = None
        getitem_2 = split_tensor[0]
        getitem_3 = split_tensor[1];  split_tensor = None
        squeeze_dim = torch.ops.aten.squeeze.dim(getitem_2, -1);  getitem_2 = None
        clone_default = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(clone_default, torch.float32)
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_2, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, amax_default, alpha = 1);  convert_element_type_default_2 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, input_41, alpha = 1);  sub_tensor_1 = input_41 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(getitem_3, -1);  getitem_3 = None
        clone_default_1 = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(clone_default_1, torch.float32)
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default_3, [1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, amax_default_1, alpha = 1);  convert_element_type_default_3 = amax_default_1 = None
        exp_default_1 = torch.ops.aten.exp.default(sub_tensor_3)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, input_42, alpha = 1);  sub_tensor_3 = input_42 = None
        return (clone_default, exp_default, sub_tensor_2, clone_default_1, exp_default_1, sub_tensor_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
