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
    def forward(self, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49):
        add_tensor = torch.ops.aten.add.Tensor(input_32, input_33, alpha = 1);  input_32 = input_33 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_34, 1.1111111111111112);  input_34 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, mul_tensor, alpha = 1);  add_tensor = mul_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(input_35, input_36);  input_35 = input_36 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, mul_tensor_2, alpha = 1);  add_tensor_1 = mul_tensor_2 = None
        view_default = torch.ops.aten.view.default(input_37, input_38);  input_37 = input_38 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_39, 0.1);  input_39 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(gt_scalar, view_default);  gt_scalar = view_default = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(mul_tensor_3, 1.1111111111111112);  mul_tensor_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, mul_tensor_4, alpha = 1);  add_tensor_2 = mul_tensor_4 = None
        pow_tensor_scalar = torch.ops.aten.pow.Tensor_Scalar(add_tensor_3, 2)
        mean_dim = torch.ops.aten.mean.dim(pow_tensor_scalar, [-1], True, dtype = None);  pow_tensor_scalar = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mean_dim, 1e-06, alpha = 1);  mean_dim = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_4);  add_tensor_4 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(add_tensor_3, rsqrt_default);  add_tensor_3 = rsqrt_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(input_40, mul_tensor_5);  input_40 = mul_tensor_5 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_41, torch.bfloat16);  input_41 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(mul_tensor_6, torch.bfloat16);  mul_tensor_6 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default, [1, 0]);  convert_element_type_default = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_1, input_42);  convert_element_type_default_1 = input_42 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default);  view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_43);  mm_default = input_43 = None
        view_default_3 = torch.ops.aten.view.default(view_default_2, input_44);  view_default_2 = input_44 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 2, 1, 3]);  view_default_3 = None
        expand_default = torch.ops.aten.expand.default(permute_default_1, input_45, implicit = False);  permute_default_1 = input_45 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_46);  clone_default = input_46 = None
        view_default_5 = torch.ops.aten.view.default(input_47, input_48);  input_47 = input_48 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_4, view_default_5);  view_default_5 = None
        view_default_6 = torch.ops.aten.view.default(bmm_default, input_49);  input_49 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_4, [0, 2, 1]);  view_default_4 = None
        return (bmm_default, view_default_6, permute_default_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
