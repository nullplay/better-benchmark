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
    def forward(self, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_33, torch.float32);  input_33 = None
        sub_tensor = torch.ops.aten.sub.Tensor(1, input_34, alpha = 1);  input_34 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, sub_tensor);  convert_element_type_default = sub_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor, torch.bfloat16);  mul_tensor = None
        mm_default = torch.ops.aten.mm.default(convert_element_type_default_1, input_35);  input_35 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_36);  permute_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mm_default, torch.float32);  mm_default = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(input_36, torch.float32);  input_36 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_3, convert_element_type_default_3);  convert_element_type_default_3 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(1, mul_tensor_1, alpha = 1);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_2, sub_tensor_1);  convert_element_type_default_2 = sub_tensor_1 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16);  mul_tensor_2 = None
        mm_default_2 = torch.ops.aten.mm.default(convert_element_type_default_4, input_37);  input_37 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_4, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_38);  permute_default_1 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0], True, dtype = torch.float32);  convert_element_type_default_4 = None
        view_default = torch.ops.aten.view.default(sum_dim_int_list, input_39);  sum_dim_int_list = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default, torch.bfloat16);  view_default = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(convert_element_type_default_5, torch.float32);  convert_element_type_default_5 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_38, torch.float32);  input_38 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(convert_element_type_default_9, convert_element_type_default_9);  convert_element_type_default_9 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(1, mul_tensor_3, alpha = 1);  mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, sub_tensor_2);  convert_element_type_default_8 = sub_tensor_2 = None
        convert_element_type_default_10 = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.bfloat16);  mul_tensor_4 = None
        mm_default_4 = torch.ops.aten.mm.default(convert_element_type_default_10, input_40);  input_40 = None
        permute_default_2 = torch.ops.aten.permute.default(convert_element_type_default_10, [1, 0])
        mm_default_5 = torch.ops.aten.mm.default(permute_default_2, input_41);  permute_default_2 = input_41 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_10, [0], True, dtype = torch.float32);  convert_element_type_default_10 = None
        view_default_1 = torch.ops.aten.view.default(sum_dim_int_list_1, input_39);  sum_dim_int_list_1 = input_39 = None
        convert_element_type_default_11 = torch.ops.prims.convert_element_type.default(mm_default_4, torch.float32);  mm_default_4 = None
        return (mm_default_1, convert_element_type_default_6, convert_element_type_default_7, mm_default_5, view_default_1, convert_element_type_default_11)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
