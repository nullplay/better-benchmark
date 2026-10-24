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
    def forward(self, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46):
        add_tensor = torch.ops.aten.add.Tensor(input_36, input_37, alpha = 1);  input_36 = input_37 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_38, input_39, alpha = 1);  input_38 = input_39 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, add_tensor, alpha = 1);  add_tensor_1 = add_tensor = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_3 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_3);  add_tensor_3 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_1, alpha = 1);  add_tensor_2 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_40);  mul_tensor = input_40 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_1, input_41, alpha = 1);  mul_tensor_1 = input_41 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_42, torch.bfloat16);  input_42 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_43, torch.bfloat16);  input_43 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_4, torch.bfloat16);  add_tensor_4 = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_44);  convert_element_type_default_2 = input_44 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(permute_default, [0, 6, 0, 0], 0);  permute_default = None
        full_default = torch.ops.aten.full.default(input_45, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_45 = None
        cat_default = torch.ops.aten.cat.default([convert_element_type_default, full_default], 0);  convert_element_type_default = full_default = None
        addmm_default = torch.ops.aten.addmm.default(cat_default, view_default, constant_pad_nd_default, beta = 1, alpha = 1);  cat_default = view_default = constant_pad_nd_default = None
        slice_tensor = torch.ops.aten.slice.Tensor(addmm_default, 1, 0, -6, 1);  addmm_default = None
        view_default_1 = torch.ops.aten.view.default(slice_tensor, input_46);  slice_tensor = input_46 = None
        split_tensor = torch.ops.aten.split.Tensor(view_default_1, 1, -1);  view_default_1 = None
        getitem_2 = split_tensor[0]
        getitem_3 = split_tensor[1];  split_tensor = None
        squeeze_dim = torch.ops.aten.squeeze.dim(getitem_2, -1);  getitem_2 = None
        clone_default = torch.ops.aten.clone.default(squeeze_dim, memory_format = torch.contiguous_format);  squeeze_dim = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(getitem_3, -1);  getitem_3 = None
        clone_default_1 = torch.ops.aten.clone.default(squeeze_dim_1, memory_format = torch.contiguous_format);  squeeze_dim_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(clone_default, torch.float32)
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_3, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, amax_default, alpha = 1);  convert_element_type_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(clone_default_1, torch.float32)
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default_4, [1], True)
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, amax_default_1, alpha = 1);  convert_element_type_default_4 = None
        return (clone_default, clone_default_1, amax_default, sub_tensor_1, amax_default_1, sub_tensor_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
