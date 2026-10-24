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
    def forward(self, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57):
        permute_default = torch.ops.aten.permute.default(input_42, [1, 0]);  input_42 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        embedding_default = torch.ops.aten.embedding.default(input_43, clone_default, -1, False, False);  clone_default = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(embedding_default, 3)
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 4);  unsqueeze_default = None
        view_default = torch.ops.aten.view.default(unsqueeze_default_1, input_44);  unsqueeze_default_1 = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default, 0);  view_default = None
        mm_default = torch.ops.aten.mm.default(squeeze_dim, input_45);  squeeze_dim = input_45 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(embedding_default, 3)
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(unsqueeze_default_2, 4);  unsqueeze_default_2 = None
        view_default_1 = torch.ops.aten.view.default(unsqueeze_default_3, input_44);  unsqueeze_default_3 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(view_default_1, 0);  view_default_1 = None
        mm_default_1 = torch.ops.aten.mm.default(squeeze_dim_1, input_46);  squeeze_dim_1 = input_46 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(embedding_default, 3);  embedding_default = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(unsqueeze_default_4, 4);  unsqueeze_default_4 = None
        view_default_2 = torch.ops.aten.view.default(unsqueeze_default_5, input_44);  unsqueeze_default_5 = input_44 = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(view_default_2, 0);  view_default_2 = None
        mm_default_2 = torch.ops.aten.mm.default(squeeze_dim_2, input_47);  squeeze_dim_2 = input_47 = None
        sub_tensor = torch.ops.aten.sub.Tensor(input_48, input_49, alpha = 1);  input_48 = input_49 = None
        rsqrt_default = torch.ops.aten.rsqrt.default(input_50);  input_50 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_51);  mul_tensor = input_51 = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_1, input_52, alpha = 1);  mul_tensor_1 = input_52 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor, torch.bfloat16);  add_tensor = None
        permute_default_1 = torch.ops.aten.permute.default(convert_element_type_default, [1, 0, 2]);  convert_element_type_default = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default_1, input_53);  clone_default_1 = input_53 = None
        permute_default_2 = torch.ops.aten.permute.default(input_43, [1, 0]);  input_43 = None
        addmm_default = torch.ops.aten.addmm.default(input_54, view_default_3, permute_default_2, beta = 1, alpha = 1);  input_54 = view_default_3 = permute_default_2 = None
        view_default_4 = torch.ops.aten.view.default(addmm_default, input_55);  addmm_default = input_55 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_56);  input_56 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_5, torch.float32);  view_default_5 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_1, [1], True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_1, amax_default, alpha = 1);  convert_element_type_default_1 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_1)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(sub_tensor_1, log_default, alpha = 1);  sub_tensor_1 = log_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(sub_tensor_2, torch.bfloat16);  sub_tensor_2 = None
        unsqueeze_default_6 = torch.ops.aten.unsqueeze.default(input_57, 1);  input_57 = None
        gather_default = torch.ops.aten.gather.default(convert_element_type_default_2, 1, unsqueeze_default_6, sparse_grad = False);  convert_element_type_default_2 = unsqueeze_default_6 = None
        squeeze_dim_3 = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        return (mm_default, mm_default_1, mm_default_2, view_default_4, squeeze_dim_3)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
