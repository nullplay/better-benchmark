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
    def forward(self, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68, input_69, input_70, input_71, input_72, input_73, input_74, input_75, input_76, input_77, input_78, input_79, input_80, input_81, input_82, input_83, input_84, input_85, input_86, input_87):
        embedding_default = torch.ops.aten.embedding.default(input_60, input_61, 1, False, False)
        mul_tensor = torch.ops.aten.mul.Tensor(embedding_default, 32.0);  embedding_default = None
        view_default = torch.ops.aten.view.default(input_62, input_63);  input_62 = None
        index_tensor = torch.ops.aten.index.Tensor(input_64, [view_default]);  input_64 = view_default = None
        view_default_1 = torch.ops.aten.view.default(index_tensor, input_65);  index_tensor = None
        add_tensor = torch.ops.aten.add.Tensor(mul_tensor, view_default_1, alpha = 1);  mul_tensor = view_default_1 = None
        addmm_default = torch.ops.aten.addmm.default(input_66, input_67, input_68, beta = 1, alpha = 1);  input_66 = input_67 = input_68 = None
        view_default_2 = torch.ops.aten.view.default(addmm_default, input_65);  addmm_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, view_default_2, alpha = 1);  add_tensor = view_default_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(add_tensor_1, torch.float32)
        var_mean_correction = torch.ops.aten.var_mean.correction(convert_element_type_default, [2], correction = 0, keepdim = True)
        sub_tensor = torch.ops.aten.sub.Tensor(convert_element_type_default, input_69, alpha = 1);  convert_element_type_default = input_69 = None
        view_default_3 = torch.ops.aten.view.default(input_70, input_65);  input_70 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, view_default_3, alpha = 1);  add_tensor_1 = view_default_3 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(add_tensor_2, torch.float32)
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, input_71, alpha = 1);  add_tensor_2 = input_71 = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_60, input_61, 1, False, False);  input_61 = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(embedding_default_1, 32.0);  embedding_default_1 = None
        view_default_4 = torch.ops.aten.view.default(input_72, input_63);  input_72 = input_63 = None
        index_tensor_1 = torch.ops.aten.index.Tensor(input_73, [view_default_4]);  input_73 = view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(index_tensor_1, input_65);  index_tensor_1 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_1, view_default_5, alpha = 1);  mul_tensor_1 = view_default_5 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_74, input_75, input_76, beta = 1, alpha = 1);  input_74 = input_75 = input_76 = None
        view_default_6 = torch.ops.aten.view.default(addmm_default_1, input_65);  addmm_default_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, view_default_6, alpha = 1);  add_tensor_4 = view_default_6 = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(add_tensor_5, torch.float32)
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(convert_element_type_default_2, [2], correction = 0, keepdim = True)
        sub_tensor_1 = torch.ops.aten.sub.Tensor(convert_element_type_default_2, input_77, alpha = 1);  convert_element_type_default_2 = input_77 = None
        view_default_7 = torch.ops.aten.view.default(input_78, input_65);  input_78 = input_65 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, view_default_7, alpha = 1);  add_tensor_5 = view_default_7 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(add_tensor_6, torch.float32)
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, input_79, alpha = 1);  add_tensor_6 = input_79 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(input_80, input_81, alpha = 1);  input_80 = input_81 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(add_tensor_8, torch.float32);  add_tensor_8 = None
        var_mean_correction_2 = torch.ops.aten.var_mean.correction(convert_element_type_default_4, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction_2[0]
        getitem_1 = var_mean_correction_2[1];  var_mean_correction_2 = None
        sub_tensor_2 = torch.ops.aten.sub.Tensor(convert_element_type_default_4, getitem_1, alpha = 1);  convert_element_type_default_4 = getitem_1 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(getitem, 1e-05, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_9);  add_tensor_9 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(sub_tensor_2, rsqrt_default);  sub_tensor_2 = rsqrt_default = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, input_82);  mul_tensor_2 = input_82 = None
        add_tensor_10 = torch.ops.aten.add.Tensor(mul_tensor_3, input_83, alpha = 1);  mul_tensor_3 = input_83 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(add_tensor_10, torch.bfloat16);  add_tensor_10 = None
        view_default_8 = torch.ops.aten.view.default(convert_element_type_default_5, input_84);  convert_element_type_default_5 = input_84 = None
        permute_default = torch.ops.aten.permute.default(input_60, [1, 0]);  input_60 = None
        mm_default = torch.ops.aten.mm.default(view_default_8, permute_default);  view_default_8 = permute_default = None
        view_default_9 = torch.ops.aten.view.default(mm_default, input_85);  mm_default = input_85 = None
        view_default_10 = torch.ops.aten.view.default(view_default_9, input_86);  input_86 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        amax_default = torch.ops.aten.amax.default(convert_element_type_default_6, [1], True)
        sub_tensor_3 = torch.ops.aten.sub.Tensor(convert_element_type_default_6, amax_default, alpha = 1);  convert_element_type_default_6 = amax_default = None
        exp_default = torch.ops.aten.exp.default(sub_tensor_3)
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [1], True, dtype = None);  exp_default = None
        log_default = torch.ops.aten.log.default(sum_dim_int_list);  sum_dim_int_list = None
        sub_tensor_4 = torch.ops.aten.sub.Tensor(sub_tensor_3, log_default, alpha = 1);  sub_tensor_3 = log_default = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(sub_tensor_4, torch.bfloat16);  sub_tensor_4 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(input_87, 1);  input_87 = None
        gather_default = torch.ops.aten.gather.default(convert_element_type_default_7, 1, unsqueeze_default, sparse_grad = False);  convert_element_type_default_7 = unsqueeze_default = None
        squeeze_dim = torch.ops.aten.squeeze.dim(gather_default, 1);  gather_default = None
        return (var_mean_correction, sub_tensor, convert_element_type_default_1, add_tensor_3, var_mean_correction_1, sub_tensor_1, convert_element_type_default_3, add_tensor_7, view_default_9, squeeze_dim)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
