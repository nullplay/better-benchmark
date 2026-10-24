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
    def forward(self, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_57, torch.float32);  input_57 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(convert_element_type_default, torch.bfloat16);  convert_element_type_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(input_58, torch.float32);  input_58 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(convert_element_type_default_2, torch.bfloat16);  convert_element_type_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(convert_element_type_default_1, convert_element_type_default_3, alpha = 1);  convert_element_type_default_1 = convert_element_type_default_3 = None
        permute_default = torch.ops.aten.permute.default(input_59, [0, 1, 4, 2, 3]);  input_59 = None
        clone_default = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default = torch.ops.aten.view.default(clone_default, input_60);  clone_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(view_default, 0);  view_default = None
        mm_default = torch.ops.aten.mm.default(input_61, squeeze_dim)
        unsqueeze_default = torch.ops.aten.unsqueeze.default(mm_default, 0);  mm_default = None
        mm_default_1 = torch.ops.aten.mm.default(squeeze_dim, input_62);  squeeze_dim = input_62 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(mm_default_1, 0);  mm_default_1 = None
        view_default_1 = torch.ops.aten.view.default(unsqueeze_default, input_63);  unsqueeze_default = None
        view_default_2 = torch.ops.aten.view.default(unsqueeze_default_1, input_64);  unsqueeze_default_1 = None
        squeeze_dim_1 = torch.ops.aten.squeeze.dim(view_default_1, 4);  view_default_1 = None
        squeeze_dim_2 = torch.ops.aten.squeeze.dim(squeeze_dim_1, 3);  squeeze_dim_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_65, input_66, alpha = 1);  input_65 = input_66 = None
        permute_default_1 = torch.ops.aten.permute.default(input_67, [0, 1, 4, 2, 3]);  input_67 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default_1, memory_format = torch.contiguous_format);  permute_default_1 = None
        view_default_3 = torch.ops.aten.view.default(clone_default_1, input_60);  clone_default_1 = None
        squeeze_dim_3 = torch.ops.aten.squeeze.dim(view_default_3, 0);  view_default_3 = None
        mm_default_2 = torch.ops.aten.mm.default(input_61, squeeze_dim_3)
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(mm_default_2, 0);  mm_default_2 = None
        mm_default_3 = torch.ops.aten.mm.default(squeeze_dim_3, input_68);  squeeze_dim_3 = input_68 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(mm_default_3, 0);  mm_default_3 = None
        view_default_4 = torch.ops.aten.view.default(unsqueeze_default_2, input_63);  unsqueeze_default_2 = None
        view_default_5 = torch.ops.aten.view.default(unsqueeze_default_3, input_64);  unsqueeze_default_3 = None
        squeeze_dim_4 = torch.ops.aten.squeeze.dim(view_default_4, 4);  view_default_4 = None
        squeeze_dim_5 = torch.ops.aten.squeeze.dim(squeeze_dim_4, 3);  squeeze_dim_4 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(input_69, torch.float32);  input_69 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_4, alpha = 1);  add_tensor_1 = convert_element_type_default_4 = None
        view_default_6 = torch.ops.aten.view.default(add_tensor, input_70);  add_tensor = input_70 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_6, [0, 1, 4, 2, 3]);  view_default_6 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_7 = torch.ops.aten.view.default(clone_default_2, input_60);  clone_default_2 = input_60 = None
        squeeze_dim_6 = torch.ops.aten.squeeze.dim(view_default_7, 0);  view_default_7 = None
        mm_default_4 = torch.ops.aten.mm.default(input_61, squeeze_dim_6);  input_61 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(mm_default_4, 0);  mm_default_4 = None
        mm_default_5 = torch.ops.aten.mm.default(squeeze_dim_6, input_71);  squeeze_dim_6 = input_71 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(mm_default_5, 0);  mm_default_5 = None
        view_default_8 = torch.ops.aten.view.default(unsqueeze_default_4, input_63);  unsqueeze_default_4 = input_63 = None
        view_default_9 = torch.ops.aten.view.default(unsqueeze_default_5, input_64);  unsqueeze_default_5 = input_64 = None
        squeeze_dim_7 = torch.ops.aten.squeeze.dim(view_default_8, 4);  view_default_8 = None
        squeeze_dim_8 = torch.ops.aten.squeeze.dim(squeeze_dim_7, 3);  squeeze_dim_7 = None
        squeeze_dim_9 = torch.ops.aten.squeeze.dim(view_default_9, 4);  view_default_9 = None
        squeeze_dim_10 = torch.ops.aten.squeeze.dim(squeeze_dim_9, 3);  squeeze_dim_9 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(squeeze_dim_8, torch.float32);  squeeze_dim_8 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(squeeze_dim_10, torch.float32);  squeeze_dim_10 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_6, alpha = 1);  add_tensor_2 = convert_element_type_default_6 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor_3, input_72);  input_72 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 1024)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_73);  mul_tensor = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(add_tensor_3, input_73);  input_73 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1], False, dtype = None);  mul_tensor_3 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0, 1], False, dtype = None);  add_tensor_3 = None
        return (view_default_2, squeeze_dim_2, view_default_5, squeeze_dim_5, convert_element_type_default_5, mul_tensor_1, sum_dim_int_list, mul_tensor_2, sum_dim_int_list_1, sum_dim_int_list_2)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
