import torch
import helion
import helion.language as hl

from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op1_op2(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 16]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(16):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        tmp9 = torch.tanh(tmp8)
        out0[tile_y, tile_x] = tmp9.permute(1, 0).to(torch.bfloat16)

def call_helion_op1_op2(buf0: torch.Tensor, arg3_1: torch.Tensor, arg4_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op1_op2(buf0, reinterpret_tensor(arg3_1, (16, 16), (1, 16), 0), arg4_1, buf2)

import torch
import helion
import helion.language as hl

from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op3_op4(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 16]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(16):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = in0[tile_x].to(torch.float32)
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        tmp9 = torch.tanh(tmp8)
        out0[tile_y, tile_x] = tmp9.permute(1, 0).to(torch.bfloat16)

def call_helion_op3_op4(buf2: torch.Tensor, arg5_1: torch.Tensor, arg6_1: torch.Tensor, buf4: torch.Tensor) -> None:
    helion_op3_op4(buf2, reinterpret_tensor(arg5_1, (16, 16), (1, 16), 0), arg6_1, buf4)

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
        tmp5 = hl.full([], 1.0, dtype=torch.float32)
        tmp6 = tmp5 * tmp4
        tmp7 = tmp5 * acc
        tmp8 = tmp6.reshape([tile_x, 1]) + tmp7.permute(1, 0)
        tmp9 = torch.tanh(tmp8)
        out0[tile_y, tile_x] = tmp9.permute(1, 0).to(torch.bfloat16)

def call_helion_op5_op6(buf4: torch.Tensor, arg7_1: torch.Tensor, arg8_1: torch.Tensor, buf6: torch.Tensor) -> None:
    helion_op5_op6(buf4, reinterpret_tensor(arg7_1, (16, 16), (1, 16), 0), arg8_1, buf6)
