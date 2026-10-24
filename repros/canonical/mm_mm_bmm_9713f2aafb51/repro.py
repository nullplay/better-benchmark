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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55):
        add_tensor = torch.ops.aten.add.Tensor(input_35, input_36, alpha = 1);  input_35 = input_36 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_37, 1.1111111111111112);  input_37 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor);  add_tensor = mul_tensor = None
        div_tensor = torch.ops.aten.div.Tensor(input_38, input_39);  input_38 = input_39 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(div_tensor, input_40);  div_tensor = input_40 = None
        neg_default = torch.ops.aten.neg.default(mul_tensor_2)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(neg_default, [2], True, dtype = torch.float32);  neg_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, mul_tensor_2, alpha = 1);  mul_tensor_1 = mul_tensor_2 = None
        where_self = torch.ops.aten.where.self(input_41, input_42, input_43);  input_41 = input_42 = input_43 = None
        mul_scalar = torch.ops.aten.mul.Scalar(where_self, 0.002607561929595828);  where_self = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_scalar, input_44);  mul_scalar = input_44 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_3, alpha = 1);  add_tensor_1 = mul_tensor_3 = None
        expand_default = torch.ops.aten.expand.default(sum_dim_int_list, input_45, implicit = False);  sum_dim_int_list = None
        div_scalar = torch.ops.aten.div.Scalar(expand_default, 768);  expand_default = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, div_scalar, alpha = 1);  add_tensor_2 = div_scalar = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_3, torch.bfloat16);  add_tensor_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_46, torch.bfloat16);  input_46 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1.1111111111111112);  convert_element_type_default_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor_4);  convert_element_type_default = mul_tensor_4 = None
        view_default = torch.ops.aten.view.default(mul_tensor_5, input_47);  mul_tensor_5 = input_47 = None
        mm_default = torch.ops.aten.mm.default(view_default, input_48);  input_48 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_49);  permute_default = input_49 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_45);  mm_default = input_45 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_50);  view_default_1 = input_50 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_2, [0, 2, 1, 3]);  view_default_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default, input_51);  clone_default = input_51 = None
        bmm_default = torch.ops.aten.bmm.default(input_52, view_default_3);  input_52 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_3, input_53);  view_default_3 = input_53 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default, input_54);  bmm_default = input_54 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default_1, input_55);  bmm_default_1 = input_55 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        return (convert_element_type_default_2, convert_element_type_default_3, permute_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
