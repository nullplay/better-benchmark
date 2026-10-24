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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43):
        mm_default = torch.ops.aten.mm.default(input_30, input_31);  input_31 = None
        permute_default = torch.ops.aten.permute.default(input_30, [1, 0]);  input_30 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_32);  permute_default = input_32 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(mm_default, input_33);  mm_default = input_33 = None
        expand_default = torch.ops.aten.expand.default(view_default, input_34, implicit = False);  view_default = input_34 = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 49);  expand_default = None
        add_tensor = torch.ops.aten.add.Tensor(input_35, input_36, alpha = 1);  input_35 = input_36 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        relu_default = torch.ops.aten.relu.default(convert_element_type_default_1);  convert_element_type_default_1 = None
        le_scalar = torch.ops.aten.le.Scalar(relu_default, 0);  relu_default = None
        full_default = torch.ops.aten.full.default(input_37, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_37 = None
        where_self = torch.ops.aten.where.self(le_scalar, full_default, div_scalar);  le_scalar = full_default = div_scalar = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(where_self, torch.float32);  where_self = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_38, 3);  input_38 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_2, [0, 2, 3], False, dtype = None)
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_39, torch.float32);  input_39 = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_3, unsqueeze_default, alpha = 1);  convert_element_type_default_3 = unsqueeze_default = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_2, sub_tensor)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 2, 3], False, dtype = None);  mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.00510204081632653)
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(mul_tensor_1, 0);  mul_tensor_1 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(input_40, 3);  input_40 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.00510204081632653)
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(input_41, 3);  input_41 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sub_tensor, unsqueeze_default_3);  sub_tensor = unsqueeze_default_3 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, mul_tensor_3, alpha = 1);  convert_element_type_default_2 = mul_tensor_3 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, unsqueeze_default_2, alpha = 1);  sub_tensor_1 = unsqueeze_default_2 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_2, input_42);  sub_tensor_2 = input_42 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, input_43);  sum_dim_int_list_1 = input_43 = None
        return (convert_element_type_default, sum_dim_int_list, unsqueeze_default_1, mul_tensor_2, mul_tensor_4, mul_tensor_5)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
