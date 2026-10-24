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
    def forward(self, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87, input_88, input_89, input_90, input_91, input_92, input_93, input_94, input_95, input_96, input_97, input_98, input_99, input_100, input_101, input_102):
        view_default = torch.ops.aten.view.default(input_74, input_75);  input_74 = input_75 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [3], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor);  add_tensor = None
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, getitem_1, alpha = 1);  convert_element_type_default = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_76);  mul_tensor = input_76 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(mul_tensor_1, input_77, alpha = 1);  mul_tensor_1 = input_77 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(input_78, torch.bfloat16);  input_78 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.bfloat16);  add_tensor_1 = None
        permute_default = torch.ops.aten.permute.default(convert_element_type_default_1, [1, 0]);  convert_element_type_default_1 = None
        view_default_1 = torch.ops.aten.view.default(convert_element_type_default_2, input_79);  convert_element_type_default_2 = input_79 = None
        mm_default = torch.ops.aten.mm.default(view_default_1, permute_default);  view_default_1 = permute_default = None
        view_default_2 = torch.ops.aten.view.default(mm_default, input_80)
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_2, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_3, [3], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem_2, 1e-05, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_3, getitem_3, alpha = 1);  convert_element_type_default_3 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_81);  mul_tensor_2 = input_81 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_3, input_82, alpha = 1);  mul_tensor_3 = input_82 = None
        view_default_3 = torch.ops.aten.view.default(add_tensor_3, input_83);  add_tensor_3 = input_83 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_3, [0, 1, 3, 2, 4, 5]);  view_default_3 = None
        view_default_4 = torch.ops.aten.view.default(input_84, input_85);  input_84 = input_85 = None
        view_default_5 = torch.ops.aten.view.default(view_default_4, input_86);  view_default_4 = input_86 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [0, 1, 3, 2, 4, 5]);  view_default_5 = None
        clone_default = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        view_default_6 = torch.ops.aten.view.default(clone_default, input_87);  clone_default = input_87 = None
        lt_scalar = torch.ops.aten.lt.Scalar(input_88, 0.9826086945831776);  input_88 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(lt_scalar, torch.bfloat16);  lt_scalar = None
        div_tensor = torch.ops.aten.div.Tensor(convert_element_type_default_4, 0.9826086945831776);  convert_element_type_default_4 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(view_default_6, div_tensor);  view_default_6 = div_tensor = None
        add_tensor_4 = torch.ops.aten.add.Tensor(view_default_2, mul_tensor_4, alpha = 1);  view_default_2 = mul_tensor_4 = None
        view_default_7 = torch.ops.aten.view.default(add_tensor_4, input_89);  add_tensor_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(view_default_7, torch.float32)
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_5, [2], correction = 0, keepdim = True)
        getitem_4 = var_mean_correction_2[0]
        getitem_5 = var_mean_correction_2[1];  var_mean_correction_2 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_4, 1e-05, alpha = 1);  getitem_4 = None
        rsqrt_default_2 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_5, getitem_5, alpha = 1);  convert_element_type_default_5 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default_2);  sub_tensor_2 = None
        mul_tensor_6 = torch.ops.aten.mul.Tensor(mul_tensor_5, input_90);  mul_tensor_5 = input_90 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_6, input_91, alpha = 1);  mul_tensor_6 = input_91 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(input_92, torch.bfloat16);  input_92 = None
        view_default_8 = torch.ops.aten.view.default(input_93, input_94);  input_93 = input_94 = None
        permute_default_3 = torch.ops.aten.permute.default(input_95, [1, 0]);  input_95 = None
        addmm_default = torch.ops.aten.addmm.default(convert_element_type_default_6, view_default_8, permute_default_3, beta = 1, alpha = 1);  convert_element_type_default_6 = view_default_8 = permute_default_3 = None
        view_default_9 = torch.ops.aten.view.default(addmm_default, input_96);  addmm_default = input_96 = None
        lt_scalar_1 = torch.ops.aten.lt.Scalar(input_97, 0.9826086945831776);  input_97 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(lt_scalar_1, torch.bfloat16);  lt_scalar_1 = None
        div_tensor_1 = torch.ops.aten.div.Tensor(convert_element_type_default_7, 0.9826086945831776);  convert_element_type_default_7 = None
        mul_tensor_7 = torch.ops.aten.mul.Tensor(view_default_9, div_tensor_1);  view_default_9 = div_tensor_1 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(view_default_7, mul_tensor_7, alpha = 1);  mul_tensor_7 = None
        view_default_10 = torch.ops.aten.view.default(add_tensor_7, input_80);  add_tensor_7 = input_80 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32)
        var_mean_correction_3 = torch.ops.aten.var_mean.correction(convert_element_type_default_8, [3], correction = 0, keepdim = True)
        getitem_6 = var_mean_correction_3[0]
        getitem_7 = var_mean_correction_3[1];  var_mean_correction_3 = None
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_8, getitem_7, alpha = 1);  convert_element_type_default_8 = None
        mul_tensor_8 = torch.ops.aten.mul.Tensor(sub_tensor_3, input_98);  sub_tensor_3 = input_98 = None
        index_tensor = torch.ops.aten.index.Tensor(input_99, [None, None, input_100]);  input_99 = input_100 = None
        div_tensor_2 = torch.ops.aten.div.Tensor(input_101, 0.9782608672976494);  input_101 = None
        mul_tensor_9 = torch.ops.aten.mul.Tensor(index_tensor, div_tensor_2);  index_tensor = div_tensor_2 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(view_default_10, mul_tensor_9, alpha = 1);  mul_tensor_9 = None
        view_default_11 = torch.ops.aten.view.default(add_tensor_8, input_89);  add_tensor_8 = input_89 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32)
        add_tensor_9 = torch.ops.aten.add.Tensor(view_default_11, input_102, alpha = 1);  input_102 = None
        return (mm_default, getitem_3, rsqrt_default_1, permute_default_1, view_default_7, getitem_5, rsqrt_default_2, add_tensor_6, view_default_10, getitem_6, getitem_7, mul_tensor_8, view_default_11, convert_element_type_default_9, add_tensor_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
