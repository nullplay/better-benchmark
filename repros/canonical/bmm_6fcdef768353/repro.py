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
    def forward(self, input_30, input_31, input_32, input_33, input_34, input_35, input_36, input_37, input_38, input_39, input_40, input_41):
        view_default = torch.ops.aten.view.default(input_30, input_31);  input_30 = input_31 = None
        permute_default = torch.ops.aten.permute.default(view_default, [2, 0, 3, 1, 4]);  view_default = None
        unbind_int = torch.ops.aten.unbind.int(permute_default, 0);  permute_default = None
        getitem = unbind_int[0]
        getitem_1 = unbind_int[1];  unbind_int = None
        mul_tensor = torch.ops.aten.mul.Tensor(getitem, 0.1767766952966369);  getitem = None
        permute_default_1 = torch.ops.aten.permute.default(getitem_1, [0, 1, 3, 2]);  getitem_1 = None
        expand_default = torch.ops.aten.expand.default(mul_tensor, input_32, implicit = False);  mul_tensor = input_32 = None
        clone_default = torch.ops.aten.clone.default(expand_default, memory_format = torch.contiguous_format);  expand_default = None
        view_default_1 = torch.ops.aten.view.default(clone_default, input_33);  clone_default = input_33 = None
        expand_default_1 = torch.ops.aten.expand.default(permute_default_1, input_34, implicit = False);  permute_default_1 = input_34 = None
        clone_default_1 = torch.ops.aten.clone.default(expand_default_1, memory_format = torch.contiguous_format);  expand_default_1 = None
        view_default_2 = torch.ops.aten.view.default(clone_default_1, input_35);  clone_default_1 = input_35 = None
        bmm_default = torch.ops.aten.bmm.default(view_default_1, view_default_2);  view_default_1 = view_default_2 = None
        view_default_3 = torch.ops.aten.view.default(bmm_default, input_36)
        view_default_4 = torch.ops.aten.view.default(input_37, input_38);  input_37 = input_38 = None
        index_tensor = torch.ops.aten.index.Tensor(input_39, [view_default_4]);  input_39 = view_default_4 = None
        view_default_5 = torch.ops.aten.view.default(index_tensor, input_40);  index_tensor = input_40 = None
        permute_default_2 = torch.ops.aten.permute.default(view_default_5, [2, 0, 1]);  view_default_5 = None
        clone_default_2 = torch.ops.aten.clone.default(permute_default_2, memory_format = torch.contiguous_format);  permute_default_2 = None
        unsqueeze_default = torch.ops.aten.unsqueeze.default(clone_default_2, 0);  clone_default_2 = None
        add_tensor = torch.ops.aten.add.Tensor(view_default_3, unsqueeze_default, alpha = 1);  view_default_3 = unsqueeze_default = None
        amax_default = torch.ops.aten.amax.default(add_tensor, [-1], True)
        sub_tensor = torch.ops.aten.sub.Tensor(add_tensor, amax_default, alpha = 1);  add_tensor = None
        exp_default = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        sum_dim_int_list = torch.ops.aten.sum.dim_IntList(exp_default, [-1], True, dtype = None)
        div_tensor = torch.ops.aten.div.Tensor(exp_default, sum_dim_int_list);  exp_default = None
        convert_element_type_default = torch.ops.prims.convert_element_type.default(div_tensor, torch.bfloat16);  div_tensor = None
        expand_default_2 = torch.ops.aten.expand.default(convert_element_type_default, input_36, implicit = False);  convert_element_type_default = input_36 = None
        view_default_6 = torch.ops.aten.view.default(expand_default_2, input_41);  expand_default_2 = input_41 = None
        return (bmm_default, amax_default, sum_dim_int_list, view_default_6)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
