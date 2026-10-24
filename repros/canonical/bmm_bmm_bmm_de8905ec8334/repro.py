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
    def forward(self, input_29, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        view_default = torch.ops.aten.view.default(input_29, input_30);  input_29 = input_30 = None
        bmm_default = torch.ops.aten.bmm.default(view_default, input_31);  view_default = input_31 = None
        view_default_1 = torch.ops.aten.view.default(bmm_default, input_32);  bmm_default = input_32 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_33, torch.float32);  input_33 = None
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default, convert_element_type_default_1);  convert_element_type_default = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(convert_element_type_default_1);  convert_element_type_default_1 = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor);  neg_default = sum_dim_int_list = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(fma_default, torch.bfloat16);  fma_default = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default_2, input_34);  convert_element_type_default_2 = input_34 = None
        bmm_default_1 = torch.ops.aten.bmm.default(input_35, view_default_2);  input_35 = None
        bmm_default_2 = torch.ops.aten.bmm.default(view_default_2, input_36);  view_default_2 = input_36 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default_1, input_37);  bmm_default_1 = input_37 = None
        view_default_4 = torch.ops.aten.view.default(bmm_default_2, input_38);  bmm_default_2 = input_38 = None
        mul_scalar = torch.ops.aten.mul.Scalar(view_default_3, 0.334370152488211);  view_default_3 = None
        permute_default = torch.ops.aten.permute.default(mul_scalar, [0, 1, 3, 2]);  mul_scalar = None
        mul_scalar_1 = torch.ops.aten.mul.Scalar(view_default_4, 0.334370152488211);  view_default_4 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [0, 2, 1, 3]);  permute_default = None
        view_default_5 = torch.ops.aten.view.default(permute_default_1, input_39);  permute_default_1 = None
        clone_default = torch.ops.aten.clone.default(view_default_5, memory_format = torch.contiguous_format);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_40);  clone_default = None
        mm_default = torch.ops.aten.mm.default(view_default_6, input_41);  input_41 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [1, 0])
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_6, [0], True, dtype = torch.float32);  view_default_6 = None
        permute_default_3 = torch.ops.aten.permute.default(mul_scalar_1, [0, 2, 1, 3]);  mul_scalar_1 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_1, input_39);  clone_default_1 = input_39 = None
        view_default_8 = torch.ops.aten.view.default(view_default_7, input_40);  view_default_7 = input_40 = None
        return (mm_default, permute_default_2, sum_dim_int_list_1, view_default_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
