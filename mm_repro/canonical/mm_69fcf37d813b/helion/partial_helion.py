import torch
import helion
import helion.language as hl

from torch._C._dynamo.guards import _reinterpret_tensor as reinterpret_tensor

@helion.kernel(static_shapes=True, dot_precision="ieee")
def helion_op0_op1(A: torch.Tensor, B: torch.Tensor, out0: torch.Tensor) -> None:
    for tile_y, tile_x in hl.tile([4096, 384]):
        acc = hl.zeros([tile_y, tile_x], dtype=torch.float32)
        for tile_r in hl.tile(512):
            tmp0 = A[tile_y, tile_r].to(torch.float32)
            tmp1 = tmp0.to(torch.bfloat16)
            tmp2 = B[tile_r, tile_x].to(torch.float32)
            tmp3 = tmp2.to(torch.bfloat16)
            acc = torch.addmm(acc, tmp1, tmp3)
        out0[tile_y.index.reshape([tile_y, 1]) // 128, (((tile_x.index.reshape([1, tile_x]) + 384*tile_y.index.reshape([tile_y, 1])) // 64) % 6), (tile_y.index.reshape([tile_y, 1]) % 128), ((tile_x.index.reshape([1, tile_x]) + 384*tile_y.index.reshape([tile_y, 1])) % 64)] = acc.to(torch.bfloat16)

def call_helion_op0_op1(arg1_1: torch.Tensor, arg0_1: torch.Tensor, buf1: torch.Tensor) -> None:
    helion_op0_op1(arg1_1, reinterpret_tensor(arg0_1, (512, 384), (1, 512), 0), buf1)
