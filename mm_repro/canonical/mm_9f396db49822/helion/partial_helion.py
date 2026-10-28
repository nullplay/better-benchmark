import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 16]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(16):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = in0[tile_y, tile_x].to(torch.float32)
        tmp6 = tmp5.to(torch.float32)
        tmp7 = tmp6 * tmp6
        tmp8 = hl.full([], 1.0, dtype=torch.float32)
        tmp9 = tmp8 - tmp7
        tmp10 = tmp4 * tmp9
        tmp11 = tmp10.to(torch.float32)
        out0[tile_y, tile_x] = tmp11.to(torch.bfloat16)

def call_helion_op0_op1(arg0_1: torch.Tensor, arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg0_1, arg1_1, arg2_1, buf1)

import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op4_op5(A: torch.Tensor, B: torch.Tensor, in0: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([128, 16]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(16):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        tmp5 = in0[tile_y, tile_x].to(torch.float32)
        tmp6 = tmp5.to(torch.float32)
        tmp7 = tmp6 * tmp6
        tmp8 = hl.full([], 1.0, dtype=torch.float32)
        tmp9 = tmp8 - tmp7
        tmp10 = tmp4 * tmp9
        tmp11 = tmp10.to(torch.float32)
        out0[tile_y, tile_x] = tmp11.to(torch.bfloat16)

def call_helion_op4_op5(buf1: torch.Tensor, arg3_1: torch.Tensor, arg4_1: torch.Tensor, buf5: torch.Tensor) -> None:
    helion_op4_op5(buf1, arg3_1, arg4_1, buf5)
