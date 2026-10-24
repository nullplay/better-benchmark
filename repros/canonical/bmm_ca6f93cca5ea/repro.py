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
    def forward(self, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69):
        add_tensor = torch.ops.aten.add.Tensor(input_46, input_47, alpha = 1);  input_46 = input_47 = None
        convolution_default = torch.ops.aten.convolution.default(input_48, input_49, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  input_48 = input_49 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convolution_default, alpha = 1);  add_tensor = convolution_default = None
        convolution_default_1 = torch.ops.aten.convolution.default(input_50, input_51, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  input_50 = input_51 = None
        view_default = torch.ops.aten.view.default(convolution_default_1, input_52);  convolution_default_1 = input_52 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0, 2, 4, 3]);  view_default = None
        unbind_int = torch.ops.aten.unbind.int(permute_default, 0);  permute_default = None
        getitem = unbind_int[2];  unbind_int = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_53, torch.float32);  input_53 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_54, input_55, alpha = 1);  input_54 = input_55 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, 0.08838834764831845);  sub_tensor = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default, alpha = 1);  convert_element_type_default = amax_default = None
        any_dims = torch.ops.aten.any.dims(input_56, [-1], True);  input_56 = None
        logical_not_default = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        where_self = torch.ops.aten.where.self(logical_not_default, mul_tensor, sub_tensor_1);  logical_not_default = mul_tensor = sub_tensor_1 = None
        exp_default = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        expand_default = torch.ops.aten.expand.default(convert_element_type_default_1, input_57, implicit = False);  convert_element_type_default_1 = input_57 = None
        view_default_1 = torch.ops.aten.view.default(expand_default, input_58);  expand_default = input_58 = None
        expand_default_1 = torch.ops.aten.expand.default(getitem, input_59, implicit = False);  getitem = None
        clone_default = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default, input_60);  clone_default = input_60 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_1, view_default_2);  view_default_1 = view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default, input_59);  bmm_default = input_59 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 1, 3, 2]);  view_default_3 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_4 = torch.ops.aten.view.default(clone_default_1, input_61);  clone_default_1 = input_61 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_62, torch.bfloat16);  input_62 = None
        convolution_default_2 = torch.ops.aten.convolution.default(view_default_4, convert_element_type_default_2, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  convert_element_type_default_2 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convolution_default_2, alpha = 1);  add_tensor_1 = convolution_default_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_2, [0, 2, 3], correction = 0, keepdim = True)
        getitem_1 = var_mean_correction[0]
        getitem_2 = var_mean_correction[1];  var_mean_correction = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(add_tensor_2, getitem_2, alpha = 1)
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_63);  sub_tensor_2 = input_63 = None
        convolution_default_3 = torch.ops.aten.convolution.default(input_64, input_65, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  input_64 = input_65 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convolution_default_3, alpha = 1);  convolution_default_3 = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_3, [0, 2, 3], correction = 0, keepdim = True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(add_tensor_3, input_66, alpha = 1);  input_66 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, input_67, alpha = 1);  input_67 = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(add_tensor_3, input_68, alpha = 1);  add_tensor_3 = input_68 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_69, 3);  input_69 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(add_tensor_2, unsqueeze_default, alpha = 1);  add_tensor_2 = unsqueeze_default = None
        return (view_default_4, getitem_1, getitem_2, mul_tensor_1, var_mean_correction_1, sub_tensor_3, add_tensor_4, sub_tensor_4, sub_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
