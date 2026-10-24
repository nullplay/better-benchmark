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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40):
        permute_default = torch.ops.aten.permute.default(input_30, [2, 0, 3, 1, 4]);  input_30 = None
        unbind_int = torch.ops.aten.unbind.int(permute_default, 0);  permute_default = None
        getitem = unbind_int[0]
        getitem_1 = unbind_int[1];  unbind_int = None
        mul_tensor = torch.ops.aten.mul.Tensor(getitem, 0.1767766952966369);  getitem = None
        permute_default_1 = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default = torch.ops.aten.expand.default(mul_tensor, input_31, implicit = False);  mul_tensor = input_31 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default = torch.ops.aten.view.default(clone_default, input_32);  clone_default = input_32 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_1, input_33, implicit = False);  permute_default_1 = input_33 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_1 = torch.ops.aten.view.default(clone_default_1, input_34);  clone_default_1 = input_34 = None
        constant_pad_nd_default = torch.ops.aten.constant_pad_nd.default(view_default, [0, 0, 0, 7, 0, 0], 0);  view_default = None
        constant_pad_nd_default_1 = torch.ops.aten.constant_pad_nd.default(view_default_1, [0, 7, 0, 0, 0, 0], 0);  view_default_1 = None
        bmm_default = torch.ops.aten.bmm.default(constant_pad_nd_default, constant_pad_nd_default_1);  constant_pad_nd_default = constant_pad_nd_default_1 = None
        slice_tensor = torch.ops.aten.slice.Tensor(bmm_default, 1, 0, -7, 1)
        slice_tensor_1 = torch.ops.aten.slice.Tensor(slice_tensor, 2, 0, -7, 1);  slice_tensor = None
        view_default_2 = torch.ops.aten.view.default(slice_tensor_1, input_35);  slice_tensor_1 = input_35 = None
        view_default_3 = torch.ops.aten.view.default(input_36, input_37);  input_36 = input_37 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_3, [2, 0, 1]);  view_default_3 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(clone_default_2, 0);  clone_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_2, unsqueeze_default, alpha = 1);  view_default_2 = unsqueeze_default = None
        view_default_4 = torch.ops.aten.view.default(add_tensor, input_38);  add_tensor = input_38 = None
        unsqueeze_default_1 = torch.ops.aten.unsqueeze.default(input_39, 1);  input_39 = None
        unsqueeze_default_2 = torch.ops.aten.unsqueeze.default(unsqueeze_default_1, 0);  unsqueeze_default_1 = None
        add_tensor_1 = torch.ops.aten.add.Tensor(view_default_4, unsqueeze_default_2, alpha = 1);  view_default_4 = unsqueeze_default_2 = None
        view_default_5 = torch.ops.aten.view.default(add_tensor_1, input_40);  add_tensor_1 = input_40 = None
        amax_default = torch.ops.aten.amax.default(view_default_5, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(view_default_5, amax_default, alpha = 1);  view_default_5 = None
        return (bmm_default, amax_default, sub_tensor)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
