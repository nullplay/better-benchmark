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
    def forward(self, input_36, input_37, input_38, input_39, input_40, input_41, input_42, input_43, input_44, input_45, input_46, input_47, input_48, input_49, input_50):
        embedding_default = torch.ops.aten.embedding.default(input_36, input_37, 3, False, False);  input_36 = input_37 = None
        embedding_default_1 = torch.ops.aten.embedding.default(input_38, input_39, -1, False, False);  input_38 = input_39 = None
        add_tensor = torch.ops.aten.add.Tensor(embedding_default, embedding_default_1, alpha = 1);  embedding_default = embedding_default_1 = None
        embedding_default_2 = torch.ops.aten.embedding.default(input_40, input_41, -1, False, False);  input_40 = input_41 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, embedding_default_2, alpha = 1);  add_tensor = embedding_default_2 = None
        var_mean_correction = torch.ops.aten.var_mean.correction(add_tensor_1, [2], correction = 0, keepdim = True)
        getitem = var_mean_correction[0]
        getitem_1 = var_mean_correction[1];  var_mean_correction = None
        add_tensor_2 = torch.ops.aten.add.Tensor(getitem, 1e-12, alpha = 1);  getitem = None
        rsqrt_default = torch.ops.aten.rsqrt.default(add_tensor_2);  add_tensor_2 = None
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor_1, getitem_1, alpha = 1);  add_tensor_1 = getitem_1 = None
        mul_tensor = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt_default);  sub_tensor = rsqrt_default = None
        mul_tensor_1 = torch.ops.aten.mul.Tensor(mul_tensor, input_42);  mul_tensor = input_42 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(mul_tensor_1, input_43, alpha = 1);  mul_tensor_1 = input_43 = None
        view_default = torch.ops.aten.view.default(add_tensor_3, input_44);  add_tensor_3 = input_44 = None
        permute_default = torch.ops.aten.permute.default(input_45, [1, 0]);  input_45 = None
        addmm_default = torch.ops.aten.addmm.default(input_46, view_default, permute_default, beta = 1, alpha = 1);  input_46 = view_default = permute_default = None
        view_default_1 = torch.ops.aten.view.default(addmm_default, input_47);  addmm_default = input_47 = None
        gt_scalar = torch.ops.aten.gt.Scalar(input_48, 0.1);  input_48 = None
        mul_tensor_2 = torch.ops.aten.mul.Tensor(gt_scalar, view_default_1);  gt_scalar = view_default_1 = None
        mul_tensor_3 = torch.ops.aten.mul.Tensor(mul_tensor_2, 1.1111111111111112);  mul_tensor_2 = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(mul_tensor_3, torch.complex64)
        _fft_c2c_default = torch.ops.aten._fft_c2c.default(convert_element_type_default, [1, 2], 0, True);  convert_element_type_default = None
        view_as_real_default = torch.ops.aten.view_as_real.default(_fft_c2c_default);  _fft_c2c_default = None
        select_int = torch.ops.aten.select.int(view_as_real_default, 3, 0);  view_as_real_default = None
        add_tensor_4 = torch.ops.aten.add.Tensor(mul_tensor_3, select_int, alpha = 1);  mul_tensor_3 = select_int = None
        var_mean_correction_1 = torch.ops.aten.var_mean.correction(add_tensor_4, [2], correction = 0, keepdim = True)
        getitem_2 = var_mean_correction_1[0]
        getitem_3 = var_mean_correction_1[1];  var_mean_correction_1 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(getitem_2, 1e-12, alpha = 1);  getitem_2 = None
        rsqrt_default_1 = torch.ops.aten.rsqrt.default(add_tensor_5);  add_tensor_5 = None
        sub_tensor_1 = torch.ops.aten.sub.Tensor(add_tensor_4, getitem_3, alpha = 1);  add_tensor_4 = getitem_3 = None
        mul_tensor_4 = torch.ops.aten.mul.Tensor(sub_tensor_1, rsqrt_default_1);  sub_tensor_1 = None
        mul_tensor_5 = torch.ops.aten.mul.Tensor(mul_tensor_4, input_49);  input_49 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(mul_tensor_5, input_50, alpha = 1);  mul_tensor_5 = input_50 = None
        div_tensor = torch.ops.aten.div.Tensor(rsqrt_default_1, 768);  rsqrt_default_1 = None
        return (mul_tensor_4, add_tensor_6, div_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
