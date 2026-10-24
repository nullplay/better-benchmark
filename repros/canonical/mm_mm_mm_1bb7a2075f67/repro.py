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
    def forward(self, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93):
        permute_default = torch.ops.aten.permute.default(input_65, [1, 0]);  input_65 = None
        add_tensor = torch.ops.aten.add.Tensor(input_66, input_67, alpha = 1);  input_66 = input_67 = None
        mul_tensor = torch.ops.aten.mul.Tensor(add_tensor, input_68);  add_tensor = input_68 = None
        view_default = torch.ops.aten.view.default(input_69, input_70);  input_69 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor, convert_element_type_default, alpha = 1);  mul_tensor = convert_element_type_default = None
        mm_default = torch.ops.aten.mm.default(input_71, input_72);  input_71 = input_72 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_70);  mm_default = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_1, alpha = 1);  add_tensor_1 = convert_element_type_default_1 = None
        view_default_2 = torch.ops.aten.view.default(input_73, input_74);  input_73 = input_74 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, input_75);  view_default_2 = input_75 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_70);  mm_default_1 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_2, alpha = 1);  add_tensor_2 = convert_element_type_default_2 = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(add_tensor_3, [0, 1], True, dtype = torch.float32)
        view_default_4 = torch.ops.aten.view.default(sum_dim_int_list, input_76);  sum_dim_int_list = None
        view_default_5 = torch.ops.aten.view.default(input_77, input_70);  input_77 = input_70 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_5, input_78, alpha = 1);  view_default_5 = input_78 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, input_79, alpha = 1);  add_tensor_4 = input_79 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(add_tensor_3, add_tensor_5);  add_tensor_5 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(add_tensor_3, input_80);  add_tensor_3 = input_80 = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1], True, dtype = torch.float32);  mul_tensor_1 = None
        view_default_6 = torch.ops.aten.view.default(sum_dim_int_list_1, input_76);  sum_dim_int_list_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mul_tensor_2, torch.bfloat16)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0], True, dtype = torch.float32)
        full_default = torch.ops.aten.full.default(input_81, True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_81 = None
        full_default_1 = torch.ops.aten.full.default(input_82, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_82 = None
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_1, full_default, [input_83], mul_tensor_2);  full_default_1 = full_default = input_83 = mul_tensor_2 = None
        slice_tensor = torch.ops.aten.slice.Tensor(input_84, 1, 0, 128, 1);  input_84 = None
        ge_scalar = torch.ops.aten.ge.Scalar(slice_tensor, 0)
        lt_scalar = torch.ops.aten.lt.Scalar(slice_tensor, 512)
        bitwise_and_tensor = torch.ops.aten.bitwise_and.Tensor(ge_scalar, lt_scalar);  ge_scalar = lt_scalar = None
        ne_scalar = torch.ops.aten.ne.Scalar(slice_tensor, -1)
        bitwise_and_tensor_1 = torch.ops.aten.bitwise_and.Tensor(bitwise_and_tensor, ne_scalar);  bitwise_and_tensor = ne_scalar = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(bitwise_and_tensor_1, -1);  bitwise_and_tensor_1 = None
        full_default_2 = torch.ops.aten.full.default(input_85, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_85 = None
        _unsafe_masked_index_put_accumulate_default_1 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_2, unsqueeze_default, [slice_tensor], sum_dim_int_list_2);  full_default_2 = unsqueeze_default = slice_tensor = sum_dim_int_list_2 = None
        view_default_7 = torch.ops.aten.view.default(convert_element_type_default_3, input_86);  convert_element_type_default_3 = input_86 = None
        mm_default_2 = torch.ops.aten.mm.default(view_default_7, input_87);  input_87 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_3 = torch.ops.aten.mm.default(permute_default_1, input_88);  permute_default_1 = input_88 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(view_default_7, [0], True, dtype = torch.float32);  view_default_7 = None
        view_default_8 = torch.ops.aten.view.default(sum_dim_int_list_3, input_76);  sum_dim_int_list_3 = input_76 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.bfloat16);  view_default_8 = None
        view_default_9 = torch.ops.aten.view.default(mm_default_2, input_89);  mm_default_2 = input_89 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_9, torch.float32);  view_default_9 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mm_default_3, torch.float32);  mm_default_3 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(convert_element_type_default_4, torch.float32);  convert_element_type_default_4 = None
        slice_tensor_1 = torch.ops.aten.slice.Tensor(convert_element_type_default_5, 2, 0, 128, 1)
        slice_tensor_2 = torch.ops.aten.slice.Tensor(convert_element_type_default_5, 2, 128, 256, 1)
        slice_tensor_3 = torch.ops.aten.slice.Tensor(convert_element_type_default_5, 2, 256, 384, 1);  convert_element_type_default_5 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(slice_tensor_3, [0, 0, -1, 0, 0, 0], 0);  slice_tensor_3 = None
        full_default_3 = torch.ops.aten.full.default(input_90, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_90 = None
        slice_scatter_default = torch.ops.aten.slice_scatter.default(full_default_3, constant_pad_nd_default, 1, 0, -1, 1);  constant_pad_nd_default = None
        add_tensor_6 = torch.ops.aten.add.Tensor(slice_tensor_2, slice_scatter_default, alpha = 1);  slice_tensor_2 = slice_scatter_default = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(slice_tensor_1, [0, 0, 0, -1, 0, 0], 0);  slice_tensor_1 = None
        slice_scatter_default_1 = torch.ops.aten.slice_scatter.default(full_default_3, constant_pad_nd_default_1, 1, 1, 9223372036854775807, 1);  full_default_3 = constant_pad_nd_default_1 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, slice_scatter_default_1, alpha = 1);  add_tensor_6 = slice_scatter_default_1 = None
        ne_scalar_1 = torch.ops.aten.ne.Scalar(input_91, 0)
        bitwise_and_tensor_2 = torch.ops.aten.bitwise_and.Tensor(input_92, ne_scalar_1);  input_92 = ne_scalar_1 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(bitwise_and_tensor_2, -1);  bitwise_and_tensor_2 = None
        full_default_4 = torch.ops.aten.full.default(input_93, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_93 = None
        _unsafe_masked_index_put_accumulate_default_2 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_4, unsqueeze_default_1, [input_91], add_tensor_7);  full_default_4 = unsqueeze_default_1 = input_91 = add_tensor_7 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(permute_default, _unsafe_masked_index_put_accumulate_default_2, alpha = 1);  permute_default = _unsafe_masked_index_put_accumulate_default_2 = None
        return (view_default_4, view_default_6, _unsafe_masked_index_put_accumulate_default, _unsafe_masked_index_put_accumulate_default_1, convert_element_type_default_6, convert_element_type_default_7, add_tensor_8)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
