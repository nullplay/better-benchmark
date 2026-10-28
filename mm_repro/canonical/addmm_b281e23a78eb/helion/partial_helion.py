import torch
import helion
import helion.language as hl

from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op2_op3(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 16]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(16):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = tmp4.to(torch.float32)
        tmp6 = hl.full([], 1.0, dtype=torch.float32)
        tmp7 = tmp6 * tmp5
        tmp8 = tmp6 * acc
        tmp9 = tmp7.reshape([tile_x, 1]) + tmp8.permute(1, 0)
        tmp10 = torch.tanh(tmp9)
        out0[tile_y, tile_x] = tmp10.permute(1, 0).to(torch.bfloat16)

def call_helion_op2_op3(buf0: torch.Tensor, buf1: torch.Tensor, arg3_1: torch.Tensor, buf3: torch.Tensor) -> None:
    helion_op2_op3(buf0, reinterpret_tensor(buf1, (16, 16), (1, 16), 0), arg3_1, buf3)

import torch
import helion
import helion.language as hl

from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op5_op6(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 16]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(16):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = tmp4.to(torch.float32)
        tmp6 = hl.full([], 1.0, dtype=torch.float32)
        tmp7 = tmp6 * tmp5
        tmp8 = tmp6 * acc
        tmp9 = tmp7.reshape([tile_x, 1]) + tmp8.permute(1, 0)
        tmp10 = torch.tanh(tmp9)
        out0[tile_y, tile_x] = tmp10.permute(1, 0).to(torch.bfloat16)

def call_helion_op5_op6(buf3: torch.Tensor, buf4: torch.Tensor, arg5_1: torch.Tensor, buf6: torch.Tensor) -> None:
    helion_op5_op6(buf3, reinterpret_tensor(buf4, (16, 16), (1, 16), 0), arg5_1, buf6)

import torch
import helion
import helion.language as hl

from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op8_op9(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 16]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(16):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = tmp4.to(torch.float32)
        tmp6 = hl.full([], 1.0, dtype=torch.float32)
        tmp7 = tmp6 * tmp5
        tmp8 = tmp6 * acc
        tmp9 = tmp7.reshape([tile_x, 1]) + tmp8.permute(1, 0)
        tmp10 = torch.tanh(tmp9)
        out0[tile_y, tile_x] = tmp10.permute(1, 0).to(torch.bfloat16)

def call_helion_op8_op9(buf6: torch.Tensor, buf7: torch.Tensor, arg7_1: torch.Tensor, buf9: torch.Tensor) -> None:
    helion_op8_op9(buf6, reinterpret_tensor(buf7, (16, 16), (1, 16), 0), arg7_1, buf9)
