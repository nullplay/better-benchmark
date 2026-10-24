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
    def forward(self, input_25, input_26, input_27, input_28, input_29, input_30, input_31, input_32, input_33, input_34):
        mul_tensor = torch.ops.aten.mul.Tensor(input_25, input_26);  input_25 = input_26 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, input_27, alpha = 1);  mul_tensor = input_27 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_28, 0.1);  input_28 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(gt_scalar, add_tensor);  gt_scalar = add_tensor = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor_1, 1.1111111111111112);  mul_tensor_1 = None
        permute_default = torch.ops.aten.permute.default(mul_tensor_2, [1, 0, 2]);  mul_tensor_2 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_29, torch.bfloat16);  input_29 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_30, torch.bfloat16);  input_30 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(clone_default, torch.bfloat16);  clone_default = None
        view_default = torch.ops.aten.view.default(convert_element_type_default_2, input_31);  convert_element_type_default_2 = input_31 = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default, view_default, permute_default_1, beta = 1, alpha = 1);  convert_element_type_default = view_default = permute_default_1 = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_32);  addmm_default = input_32 = None
        view_default_2 = torch.ops.aten.view.default(view_default_1, input_33);  input_33 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32);  view_default_2 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_3, [1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default_3, amax_default, alpha = 1);  convert_element_type_default_3 = None
        exp_default = torch.ops.aten.exp.default(sub_tensor)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, log_default, alpha = 1);  sub_tensor = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(sub_tensor_1, torch.bfloat16);  sub_tensor_1 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        gather_default = torch.ops.aten.gather.default(convert_element_type_default_5, 1, input_34, sparse_grad = False);  convert_element_type_default_5 = input_34 = None
        return (view_default_1, amax_default, log_default, gather_default)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
