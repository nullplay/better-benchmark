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
    def forward(self, input_16, input_17, input_18, input_19, input_20, input_21, input_22, input_23, input_24, input_25, input_26, input_27, input_28):
        convolution_default = torch.ops.aten.convolution.default(input_16, input_17, input_18, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  input_16 = input_17 = input_18 = None
        relu_default = torch.ops.aten.relu.default(convolution_default);  convolution_default = None
        convolution_default_1 = torch.ops.aten.convolution.default(relu_default, input_19, input_20, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_default = input_19 = input_20 = None
        relu_default_1 = torch.ops.aten.relu.default(convolution_default_1);  convolution_default_1 = None
        _low_memory_max_pool_with_offsets_default = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu_default_1, [3, 3], [2, 2], [0, 0], [1, 1], False);  relu_default_1 = None
        getitem = _low_memory_max_pool_with_offsets_default[0];  _low_memory_max_pool_with_offsets_default = None
        _adaptive_avg_pool2d_default = torch.ops.aten._adaptive_avg_pool2d.default(getitem, input_21);  getitem = input_21 = None
        view_default = torch.ops.aten.view.default(_adaptive_avg_pool2d_default, input_22);  _adaptive_avg_pool2d_default = input_22 = None
        permute_default = torch.ops.aten.permute.default(input_23, [1, 0]);  input_23 = None
        addmm_default = torch.ops.aten.addmm.default(input_24, view_default, permute_default, beta = 1, alpha = 1);  input_24 = view_default = permute_default = None
        relu_default_2 = torch.ops.aten.relu.default(addmm_default);  addmm_default = None
        permute_default_1 = torch.ops.aten.permute.default(input_25, [1, 0]);  input_25 = None
        addmm_default_1 = torch.ops.aten.addmm.default(input_26, relu_default_2, permute_default_1, beta = 1, alpha = 1);  input_26 = relu_default_2 = permute_default_1 = None
        relu_default_3 = torch.ops.aten.relu.default(addmm_default_1);  addmm_default_1 = None
        permute_default_2 = torch.ops.aten.permute.default(input_27, [1, 0]);  input_27 = None
        addmm_default_2 = torch.ops.aten.addmm.default(input_28, relu_default_3, permute_default_2, beta = 1, alpha = 1);  input_28 = relu_default_3 = permute_default_2 = None
        return (addmm_default_2,)


def make_inputs(shape_config=None):
    if shape_config is None:
        shape_config = default_shape_config(__file__)
    if shape_config is None:
        raise RuntimeError("No input points in the sibling shapes.json")
    return make_inputs_from_config(shape_config)


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
