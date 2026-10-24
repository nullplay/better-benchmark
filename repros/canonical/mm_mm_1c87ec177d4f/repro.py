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
    def forward(self, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57):
        mm_default = torch.ops.aten.mm.default(input_38, input_39);  input_39 = None
        permute_default = torch.ops.aten.permute.default(input_38, [1, 0]);  input_38 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_40);  permute_default = input_40 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(mm_default, input_41);  mm_default = input_41 = None
        expand_default = torch.ops.aten.expand.default(view_default, input_42, implicit = False);  view_default = input_42 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        full_default = torch.ops.aten.full.default(input_43, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_43 = None
        where_self = torch.ops.aten.where.self(input_44, full_default, div_scalar);  input_44 = div_scalar = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(where_self, torch.float32)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [0, 2, 3], False, dtype = None)
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_45, torch.float32);  input_45 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_2, input_46, alpha = 1);  convert_element_type_default_2 = input_46 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, sub_tensor)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3], False, dtype = None);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00015943877551020407)
        unsqueeze_default = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_47, 3);  input_47 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00015943877551020407)
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(input_48, 3);  input_48 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_2);  sub_tensor = unsqueeze_default_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_tensor_3, alpha = 1);  convert_element_type_default_1 = mul_tensor_3 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_1, alpha = 1);  sub_tensor_1 = unsqueeze_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_49);  sub_tensor_2 = input_49 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, input_50);  sum_dim_int_list_1 = input_50 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_51, torch.bfloat16);  input_51 = None
        convolution_backward_default = torch.ops.aten.convolution_backward.default(convert_element_type_default_3, input_52, input_53, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_default_3 = input_53 = None
        getitem = convolution_backward_default[0];  convolution_backward_default = None
        add_tensor = torch.ops.aten.add.Tensor(where_self, getitem, alpha = 1);  where_self = getitem = None
        le_scalar = torch.ops.aten.le.Scalar(input_52, 0);  input_52 = None
        where_self_1 = torch.ops.aten.where.self(le_scalar, full_default, add_tensor);  le_scalar = add_tensor = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(where_self_1, torch.float32)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 2, 3], False, dtype = None)
        mul_tensor_6 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, input_54);  input_54 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, input_55, alpha = 1);  convert_element_type_default_4 = input_55 = None
        getitem_1 = input_56[0];  input_56 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(where_self_1, getitem_1, alpha = 1);  where_self_1 = getitem_1 = None
        where_self_2 = torch.ops.aten.where.self(input_57, full_default, add_tensor_1);  input_57 = full_default = add_tensor_1 = None
        return (convert_element_type_default, sum_dim_int_list, unsqueeze_default, mul_tensor_2, mul_tensor_4, mul_tensor_5, sum_dim_int_list_2, mul_tensor_6, sub_tensor_3, where_self_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
