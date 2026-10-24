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
    def forward(self, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71):
        full_default = torch.ops.aten.full.default(input_48, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_48 = None
        mul_tensor = torch.ops.aten.mul.Tensor(input_49, input_50);  input_49 = input_50 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, 768)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(mul_tensor, [2], True, dtype = None)
        mul_tensor_2 = torch.ops.aten.mul.Tensor(mul_tensor, input_51);  mul_tensor = None
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [2], True, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(input_51, sum_dim_int_list_1);  input_51 = sum_dim_int_list_1 = None
        sub_tensor = torch.ops.aten.sub.Tensor(mul_tensor_1, sum_dim_int_list, alpha = 1);  mul_tensor_1 = sum_dim_int_list = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(sub_tensor, mul_tensor_3, alpha = 1);  sub_tensor = mul_tensor_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(input_52, sub_tensor_1);  input_52 = sub_tensor_1 = None
        select_scatter_default = torch.ops.aten.select_scatter.default(full_default, mul_tensor_4, 3, 0);  full_default = None
        view_as_complex_default = torch.ops.aten.view_as_complex.default(select_scatter_default);  select_scatter_default = None
        _fft_c2c_default = torch.ops.aten._fft_c2c.default(view_as_complex_default, [1, 2], 0, False);  view_as_complex_default = None
        view_as_real_default = torch.ops.aten.view_as_real.default(_fft_c2c_default);  _fft_c2c_default = None
        select_int = torch.ops.aten.select.int(view_as_real_default, 3, 0);  view_as_real_default = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor_4, select_int, alpha = 1);  mul_tensor_4 = select_int = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_53, torch.float32);  input_53 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(add_tensor, mul_tensor_5);  add_tensor = mul_tensor_5 = None
        view_default = torch.ops.aten.view.default(mul_tensor_6, input_54);  mul_tensor_6 = input_54 = None
        permute_default = torch.ops.aten.permute.default(input_55, [1, 0]);  input_55 = None
        permute_default_1 = torch.ops.aten.permute.default(permute_default, [1, 0]);  permute_default = None
        mm_default = torch.ops.aten.mm.default(view_default, permute_default_1);  permute_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default, [1, 0]);  view_default = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default_2, input_56);  permute_default_2 = input_56 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_57);  mm_default = input_57 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_1, input_58);  input_58 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(mul_tensor_7, 768)
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [2], True, dtype = None)
        mul_tensor_9 = torch.ops.aten.mul.Tensor(mul_tensor_7, input_59);  mul_tensor_7 = None
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [2], True, dtype = None);  mul_tensor_9 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(input_59, sum_dim_int_list_3);  sum_dim_int_list_3 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(mul_tensor_8, sum_dim_int_list_2, alpha = 1);  mul_tensor_8 = sum_dim_int_list_2 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, mul_tensor_10, alpha = 1);  sub_tensor_2 = mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(input_60, sub_tensor_3);  input_60 = sub_tensor_3 = None
        mul_tensor_12 = torch.ops.aten.mul.Tensor(view_default_1, input_59);  input_59 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1], False, dtype = None);  mul_tensor_12 = None
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(view_default_1, [0, 1], False, dtype = None);  view_default_1 = None
        sum_dim_int_list_6 = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0], True, dtype = None)
        _unsafe_masked_index_put_accumulate_default = torch.ops.aten._unsafe_masked_index_put_accumulate.default(input_61, input_62, [input_63], sum_dim_int_list_6);  input_61 = input_62 = input_63 = sum_dim_int_list_6 = None
        expand_default = torch.ops.aten.expand.default(input_64, input_65, implicit = False);  input_64 = input_65 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_66, -1);  input_66 = None
        full_default_1 = torch.ops.aten.full.default(input_67, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_67 = None
        _unsafe_masked_index_put_accumulate_default_1 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_1, unsqueeze_default, [expand_default], mul_tensor_11);  full_default_1 = unsqueeze_default = expand_default = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_68, -1);  input_68 = None
        full_default_2 = torch.ops.aten.full.default(input_69, 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_69 = None
        _unsafe_masked_index_put_accumulate_default_2 = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_2, unsqueeze_default_1, [input_70], mul_tensor_11);  full_default_2 = unsqueeze_default_1 = input_70 = mul_tensor_11 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(input_71, _unsafe_masked_index_put_accumulate_default_2, alpha = 1);  input_71 = _unsafe_masked_index_put_accumulate_default_2 = None
        return (mm_default_1, sum_dim_int_list_4, sum_dim_int_list_5, _unsafe_masked_index_put_accumulate_default, _unsafe_masked_index_put_accumulate_default_1, add_tensor_1)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
