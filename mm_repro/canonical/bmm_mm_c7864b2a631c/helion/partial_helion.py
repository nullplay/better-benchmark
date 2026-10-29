import torch
import helion
import helion.language as hl

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op0_op2(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 2048]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(2048):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        tmp4 = acc.to(torch.float32)
        out0[tile_y.index.reshape([tile_y, 1]) // 128, (((tile_x.index.reshape([1, tile_x]) + 2048*tile_y.index.reshape([tile_y, 1])) // 128) % 16), (tile_y.index.reshape([tile_y, 1]) % 128), ((tile_x.index.reshape([1, tile_x]) + 2048*tile_y.index.reshape([tile_y, 1])) % 128)] = tmp4.to(torch.float32)

def call_helion_op0_op2(arg1_1: torch.Tensor, arg2_1: torch.Tensor, buf2: torch.Tensor) -> None:
    helion_op0_op2(arg1_1, arg2_1, buf2)
