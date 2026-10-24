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
    def forward(self, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54):
        mul_tensor = torch.ops.aten.mul.Tensor(input_38, input_39);  input_38 = input_39 = None
        view_default = torch.ops.aten.view.default(input_40, input_41);  input_40 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default, input_42, alpha = 1);  view_default = input_42 = None
        full_default = torch.ops.aten.full.default(input_43, -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        div_tensor = torch.ops.aten.div.Tensor(input_44, input_45);  input_44 = input_45 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, div_tensor);  mul_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [-1], True, dtype = None)
        neg_default = torch.ops.aten.neg.default(div_tensor);  div_tensor = None
        fma_default = torch.ops.prims.fma.default(neg_default, sum_dim_int_list, mul_tensor_1);  neg_default = sum_dim_int_list = mul_tensor_1 = None
        view_default_1 = torch.ops.aten.view.default(fma_default, input_41);  fma_default = input_41 = None
        div_scalar = torch.ops.aten.div.Scalar(view_default_1, 2)
        eq_tensor = torch.ops.aten.eq.Tensor(add_tensor, full_default)
        where_self = torch.ops.aten.where.self(eq_tensor, div_scalar, view_default_1);  eq_tensor = div_scalar = view_default_1 = None
        lt_tensor = torch.ops.aten.lt.Tensor(add_tensor, full_default);  add_tensor = full_default = None
        full_default_1 = torch.ops.aten.full.default(input_43, 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_43 = None
        where_self_1 = torch.ops.aten.where.self(lt_tensor, full_default_1, where_self);  lt_tensor = full_default_1 = where_self = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(where_self_1, torch.bfloat16);  where_self_1 = None
        view_default_2 = torch.ops.aten.view.default(convert_element_type_default, input_46);  convert_element_type_default = input_46 = None
        bmm_default = torch.ops.aten.bmm.default(input_47, view_default_2);  input_47 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_2, input_48);  view_default_2 = input_48 = None
        permute_default = torch.ops.aten.permute.default(bmm_default, [0, 2, 1]);  bmm_default = None
        view_default_3 = torch.ops.aten.view.default(permute_default, input_49);  permute_default = None
        view_default_4 = torch.ops.aten.view.default(bmm_default_1, input_49);  bmm_default_1 = input_49 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1, 3]);  view_default_4 = None
        clone_default = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_5 = torch.ops.aten.view.default(clone_default, input_50);  clone_default = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        view_default_6 = torch.ops.aten.view.default(permute_default_2, input_50);  permute_default_2 = None
        clone_default_1 = torch.ops.aten.clone.default(view_default_6, memory_format = torch.contiguous_format);  view_default_6 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_1, input_51);  clone_default_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(view_default_5, 0.125);  view_default_5 = None
        view_default_8 = torch.ops.aten.view.default(mul_tensor_2, input_51);  mul_tensor_2 = input_51 = None
        mm_default = torch.ops.aten.mm.default(view_default_8, input_52);  input_52 = None
        permute_default_3 = torch.ops.aten.permute.default(view_default_8, [1, 0])
        mm_default_1 = torch.ops.aten.mm.default(permute_default_3, input_53);  permute_default_3 = input_53 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(view_default_8, [0], True, dtype = torch.float32);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(sum_dim_int_list_1, input_54);  sum_dim_int_list_1 = input_54 = None
        view_default_10 = torch.ops.aten.view.default(mm_default, input_50);  mm_default = input_50 = None
        return (view_default_7, mm_default_1, view_default_9, view_default_10)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
