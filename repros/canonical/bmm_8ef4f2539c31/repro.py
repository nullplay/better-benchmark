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
    def forward(self, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42):
        view_default = torch.ops.aten.view.default(input_35, input_36);  input_35 = input_36 = None
        permute_default = torch.ops.aten.permute.default(view_default, [1, 0, 2, 4, 3]);  view_default = None
        unbind_int = torch.ops.aten.unbind.int(permute_default, 0);  permute_default = None
        getitem = unbind_int[0]
        getitem_1 = unbind_int[1];  unbind_int = None
        permute_default_1 = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default = torch.ops.aten.expand.default(getitem, input_37, implicit = False);  getitem = input_37 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_38);  clone_default = input_38 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_1, input_39, implicit = False);  permute_default_1 = input_39 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default_1, input_40);  clone_default_1 = input_40 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default_1, [0, 0, 0, 7, 0, 0], 0);  view_default_1 = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(view_default_2, [0, 7, 0, 0, 0, 0], 0);  view_default_2 = None
        bmm_default = torch.ops.aten.bmm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        slice_tensor = torch.ops.aten.slice.Tensor(bmm_default, 1, 0, -7, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(slice_tensor, 2, 0, -7, 1);  slice_tensor = None
        view_default_3 = torch.ops.aten.view.default(slice_tensor_1, input_41);  slice_tensor_1 = input_41 = None
        mul_tensor = torch.ops.aten.mul.Tensor(view_default_3, 0.08838834764831845)
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default_1, 1);  convert_element_type_default_1 = None
        amax_default = torch.ops.aten.amax.default(mul_tensor_1, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default, alpha = 1);  mul_tensor_1 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor, 0.08838834764831845);  sub_tensor = None
        amax_default_1 = torch.ops.aten.amax.default(convert_element_type_default, [-1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default, amax_default_1, alpha = 1)
        abs_default = torch.ops.aten.abs.default(convert_element_type_default)
        ne_scalar = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        eq_tensor = torch.ops.aten.eq.Tensor(convert_element_type_default, convert_element_type_default);  convert_element_type_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default = torch.ops.aten.logical_not.default(mul_tensor_3);  mul_tensor_3 = None
        logical_not_default_1 = torch.ops.aten.logical_not.default(input_42);  input_42 = None
        where_self = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  logical_not_default_1 = mul_tensor_2 = sub_tensor_1 = None
        exp_default = torch.ops.aten.exp.default(where_self);  where_self = None
        return (bmm_default, amax_default, amax_default_1, logical_not_default, exp_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
