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
    def forward(self, input_45, input_46, input_47, input_48, input_49, input_50, input_51, input_52, input_53, input_54, input_55, input_56, input_57, input_58, input_59, input_60, input_61, input_62, input_63, input_64, input_65, input_66, input_67, input_68):
        convert_element_type_default = torch.ops.prims.convert_element_type.default(input_45, torch.float32);  input_45 = None
        add_tensor = torch.ops.aten.add.Tensor(input_46, convert_element_type_default, alpha = 1);  input_46 = convert_element_type_default = None
        view_default = torch.ops.aten.view.default(input_47, input_48);  input_47 = None
        convert_element_type_default_1 = torch.ops.prims.convert_element_type.default(view_default, torch.float32);  view_default = None
        add_tensor_1 = torch.ops.aten.add.Tensor(add_tensor, convert_element_type_default_1, alpha = 1);  add_tensor = convert_element_type_default_1 = None
        mm_default = torch.ops.aten.mm.default(input_49, input_50);  input_49 = input_50 = None
        view_default_1 = torch.ops.aten.view.default(mm_default, input_48);  mm_default = None
        convert_element_type_default_2 = torch.ops.prims.convert_element_type.default(view_default_1, torch.float32);  view_default_1 = None
        add_tensor_2 = torch.ops.aten.add.Tensor(add_tensor_1, convert_element_type_default_2, alpha = 1);  add_tensor_1 = convert_element_type_default_2 = None
        view_default_2 = torch.ops.aten.view.default(input_51, input_52);  input_51 = None
        mm_default_1 = torch.ops.aten.mm.default(view_default_2, input_53);  view_default_2 = input_53 = None
        view_default_3 = torch.ops.aten.view.default(mm_default_1, input_48);  mm_default_1 = None
        convert_element_type_default_3 = torch.ops.prims.convert_element_type.default(view_default_3, torch.float32);  view_default_3 = None
        add_tensor_3 = torch.ops.aten.add.Tensor(add_tensor_2, convert_element_type_default_3, alpha = 1);  add_tensor_2 = convert_element_type_default_3 = None
        clone_default = torch.ops.aten.clone.default(input_54, memory_format = torch.contiguous_format);  input_54 = None
        view_default_4 = torch.ops.aten.view.default(clone_default, input_55);  clone_default = input_55 = None
        bmm_default = torch.ops.aten.bmm.default(input_56, view_default_4);  input_56 = None
        bmm_default_1 = torch.ops.aten.bmm.default(view_default_4, input_57);  view_default_4 = input_57 = None
        view_default_5 = torch.ops.aten.view.default(bmm_default, input_58);  bmm_default = input_58 = None
        permute_default = torch.ops.aten.permute.default(view_default_5, [0, 2, 1, 3]);  view_default_5 = None
        clone_default_1 = torch.ops.aten.clone.default(permute_default, memory_format = torch.contiguous_format);  permute_default = None
        view_default_6 = torch.ops.aten.view.default(clone_default_1, input_59);  clone_default_1 = input_59 = None
        view_default_7 = torch.ops.aten.view.default(view_default_6, input_52);  view_default_6 = None
        permute_default_1 = torch.ops.aten.permute.default(view_default_7, [1, 0])
        mm_default_2 = torch.ops.aten.mm.default(permute_default_1, input_60);  permute_default_1 = input_60 = None
        mm_default_3 = torch.ops.aten.mm.default(view_default_7, input_61);  view_default_7 = input_61 = None
        view_default_8 = torch.ops.aten.view.default(mm_default_3, input_48);  mm_default_3 = None
        convert_element_type_default_4 = torch.ops.prims.convert_element_type.default(view_default_8, torch.float32);  view_default_8 = None
        add_tensor_4 = torch.ops.aten.add.Tensor(add_tensor_3, convert_element_type_default_4, alpha = 1);  add_tensor_3 = convert_element_type_default_4 = None
        convert_element_type_default_5 = torch.ops.prims.convert_element_type.default(mm_default_2, torch.float32);  mm_default_2 = None
        view_default_9 = torch.ops.aten.view.default(input_62, input_52);  input_62 = input_52 = None
        mm_default_4 = torch.ops.aten.mm.default(view_default_9, input_63);  view_default_9 = input_63 = None
        view_default_10 = torch.ops.aten.view.default(mm_default_4, input_48);  mm_default_4 = None
        convert_element_type_default_6 = torch.ops.prims.convert_element_type.default(view_default_10, torch.float32);  view_default_10 = None
        add_tensor_5 = torch.ops.aten.add.Tensor(add_tensor_4, convert_element_type_default_6, alpha = 1);  add_tensor_4 = convert_element_type_default_6 = None
        mm_default_5 = torch.ops.aten.mm.default(input_64, input_65);  input_64 = input_65 = None
        view_default_11 = torch.ops.aten.view.default(mm_default_5, input_48);  mm_default_5 = None
        convert_element_type_default_7 = torch.ops.prims.convert_element_type.default(view_default_11, torch.float32);  view_default_11 = None
        add_tensor_6 = torch.ops.aten.add.Tensor(add_tensor_5, convert_element_type_default_7, alpha = 1);  add_tensor_5 = convert_element_type_default_7 = None
        view_default_12 = torch.ops.aten.view.default(input_66, input_48);  input_66 = input_48 = None
        convert_element_type_default_8 = torch.ops.prims.convert_element_type.default(view_default_12, torch.float32);  view_default_12 = None
        add_tensor_7 = torch.ops.aten.add.Tensor(add_tensor_6, convert_element_type_default_8, alpha = 1);  add_tensor_6 = convert_element_type_default_8 = None
        convert_element_type_default_9 = torch.ops.prims.convert_element_type.default(input_67, torch.float32);  input_67 = None
        add_tensor_8 = torch.ops.aten.add.Tensor(add_tensor_7, convert_element_type_default_9, alpha = 1);  add_tensor_7 = convert_element_type_default_9 = None
        add_tensor_9 = torch.ops.aten.add.Tensor(add_tensor_8, input_68, alpha = 1);  add_tensor_8 = input_68 = None
        return (bmm_default_1, convert_element_type_default_5, add_tensor_9)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
