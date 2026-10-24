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
    def forward(self, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_60, torch.float32);  input_60 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_61, torch.float32);  input_61 = None
        sub_tensor = torch.ops.aten.sub.Tensor(1, convert_element_type_default_1, alpha = 1)
        mul_tensor = torch.ops.aten.mul.Tensor(convert_element_type_default_1, sub_tensor);  convert_element_type_default_1 = sub_tensor = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(convert_element_type_default, mul_tensor);  convert_element_type_default = mul_tensor = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(mul_tensor_1, torch.bfloat16);  mul_tensor_1 = None
        mm_default = torch.ops.aten.mm.default(convert_element_type_default_2, input_62);  input_62 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_2, [1, 0]);  convert_element_type_default_2 = None
        mm_default_1 = torch.ops.aten.mm.default(permute_default, input_63);  permute_default = input_63 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(mm_default_1, torch.float32);  mm_default_1 = None
        view_default = torch.ops.aten.view.default(mm_default, input_64);  mm_default = input_64 = None
        avg_pool2d_backward_default = torch.ops.aten.avg_pool2d_backward.default(view_default, input_65, [4, 4], [], [0, 0], False, True, None);  view_default = None
        le_scalar = torch.ops.aten.le.Scalar(input_65, 0);  input_65 = None
        full_default = torch.ops.aten.full.default(input_66, 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  input_66 = None
        where_self = torch.ops.aten.where.self(le_scalar, full_default, avg_pool2d_backward_default);  le_scalar = avg_pool2d_backward_default = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(where_self, torch.float32)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(convert_element_type_default_4, [0, 2, 3], False, dtype = None)
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(input_67, torch.float32);  input_67 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, input_68, alpha = 1);  convert_element_type_default_5 = input_68 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(convert_element_type_default_4, sub_tensor_1)
        sum_dim_int_list_1 = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 2, 3], False, dtype = None);  mul_tensor_2 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(sum_dim_int_list, 0.0006510416666666666)
        unsqueeze_default = torch.ops.aten.unsqueeze.default(mul_tensor_3, 0);  mul_tensor_3 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(unsqueeze_default, 2);  unsqueeze_default = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 3);  unsqueeze_default_1 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, 0.0006510416666666666)
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_69);  mul_tensor_4 = input_69 = None
        unsqueeze_default_3 = torch.ops.aten.unsqueeze.default(input_70, 2);  input_70 = None
        unsqueeze_default_4 = torch.ops.aten.unsqueeze.default(unsqueeze_default_3, 3);  unsqueeze_default_3 = None
        unsqueeze_default_5 = torch.ops.aten.unsqueeze.default(input_71, 3);  input_71 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(sub_tensor_1, unsqueeze_default_4);  sub_tensor_1 = unsqueeze_default_4 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, mul_tensor_6, alpha = 1);  convert_element_type_default_4 = mul_tensor_6 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(sub_tensor_2, unsqueeze_default_2, alpha = 1);  sub_tensor_2 = unsqueeze_default_2 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(sub_tensor_3, unsqueeze_default_5);  sub_tensor_3 = unsqueeze_default_5 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sum_dim_int_list_1, input_72);  sum_dim_int_list_1 = input_72 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(mul_tensor_7, torch.bfloat16);  mul_tensor_7 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(input_73, input_74);  input_73 = input_74 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(mul_tensor_9, torch.bfloat16);  mul_tensor_9 = None
        convolution_backward_default = torch.ops.aten.convolution_backward.default(convert_element_type_default_7, input_75, input_76, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  convert_element_type_default_7 = input_76 = None
        getitem = convolution_backward_default[0];  convolution_backward_default = None
        add_tensor = torch.ops.aten.add.Tensor(where_self, getitem, alpha = 1);  where_self = getitem = None
        le_scalar_1 = torch.ops.aten.le.Scalar(input_75, 0);  input_75 = None
        where_self_1 = torch.ops.aten.where.self(le_scalar_1, full_default, add_tensor);  le_scalar_1 = full_default = add_tensor = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(where_self_1, torch.float32);  where_self_1 = None
        sum_dim_int_list_2 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_8, [0, 2, 3], False, dtype = None)
        sub_tensor_4 = torch.ops.aten.sub.Tensor(input_77, input_78, alpha = 1);  input_77 = input_78 = None
        mul_tensor_10 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, sub_tensor_4)
        sum_dim_int_list_3 = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 2, 3], False, dtype = None);  mul_tensor_10 = None
        mul_tensor_11 = torch.ops.aten.mul.Tensor(sum_dim_int_list_2, 0.0006510416666666666)
        mul_tensor_12 = torch.ops.aten.mul.Tensor(sub_tensor_4, input_79);  sub_tensor_4 = input_79 = None
        sub_tensor_5 = torch.ops.aten.sub.Tensor(convert_element_type_default_8, mul_tensor_12, alpha = 1);  mul_tensor_12 = None
        sub_tensor_6 = torch.ops.aten.sub.Tensor(sub_tensor_5, input_80, alpha = 1);  sub_tensor_5 = input_80 = None
        sum_dim_int_list_4 = torch.ops.aten.sum.dim_IntList(convert_element_type_default_8, [0, 2, 3], False, dtype = None)
        sub_tensor_7 = torch.ops.aten.sub.Tensor(input_81, input_82, alpha = 1);  input_81 = input_82 = None
        mul_tensor_13 = torch.ops.aten.mul.Tensor(convert_element_type_default_8, sub_tensor_7)
        sum_dim_int_list_5 = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 2, 3], False, dtype = None);  mul_tensor_13 = None
        mul_tensor_14 = torch.ops.aten.mul.Tensor(sum_dim_int_list_4, 0.0006510416666666666)
        mul_tensor_15 = torch.ops.aten.mul.Tensor(sub_tensor_7, input_83);  sub_tensor_7 = input_83 = None
        sub_tensor_8 = torch.ops.aten.sub.Tensor(convert_element_type_default_8, mul_tensor_15, alpha = 1);  convert_element_type_default_8 = mul_tensor_15 = None
        sub_tensor_9 = torch.ops.aten.sub.Tensor(sub_tensor_8, input_84, alpha = 1);  sub_tensor_8 = input_84 = None
        return (convert_element_type_default_3, sum_dim_int_list, mul_tensor_5, mul_tensor_8, convert_element_type_default_6, sum_dim_int_list_2, sum_dim_int_list_3, mul_tensor_11, sub_tensor_6, sum_dim_int_list_4, sum_dim_int_list_5, mul_tensor_14, sub_tensor_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
