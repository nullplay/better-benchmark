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
    def forward(self, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_27, torch.bfloat16);  input_27 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.float32);  convert_element_type_default = None
        exp_default = torch.ops.aten.exp.default(input_28);  input_28 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_1, [1], True, dtype = None)
        mul_tensor = torch.ops.aten.mul.Tensor(exp_default, sum_dim_int_list);  exp_default = sum_dim_int_list = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_1, mul_tensor, alpha = 1);  convert_element_type_default_1 = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(sub_tensor, torch.bfloat16);  sub_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(input_29, convert_element_type_default_2, alpha = 1);  input_29 = convert_element_type_default_2 = None
        full_default = torch.ops.aten.full.default(input_30, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_30 = None
        index_put_default = torch.ops.aten.index_put.default(full_default, [input_31, input_32], add_tensor, True);  full_default = input_31 = input_32 = add_tensor = None
        view_default = torch.ops.aten.view.default(index_put_default, input_33);  index_put_default = input_33 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0])
        mm_default = torch.ops.aten.mm.default(permute_default, input_34);  permute_default = input_34 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default, input_35);  view_default = input_35 = None
        view_default_1 = torch.ops.aten.view.default(mm_default_1, input_36);  mm_default_1 = input_36 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_37);  input_37 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 768)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [2], True, dtype = None)
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_1, input_38);  mul_tensor_1 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [2], True, dtype = None);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_38, sum_dim_int_list_2);  sum_dim_int_list_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(mul_tensor_2, sum_dim_int_list_1, alpha = 1);  mul_tensor_2 = sum_dim_int_list_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, input_38);  input_38 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1], False, dtype = None);  mul_tensor_5 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_3, [0, 1], False, dtype = None);  convert_element_type_default_3 = None
        return (convert_element_type_default_4, mul_tensor_4, sub_tensor_1, sum_dim_int_list_3, sum_dim_int_list_4)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
