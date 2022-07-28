import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from flash_attn.flash_attention import FlashMHA

class OriginMha(nn.Module):
    def __init__(self, head_dim = 12):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.scale = head_dim ** -0.5

    def forward(self, q, k ,v, mask = None, pos_bias = None):
        q = q * self.scale
        S = (q @ k.transpose(-2, -1))

        if mask is not None:
            S = S + mask.unsqueeze(1).unsqueeze(0)

        P = self.softmax(S)

        O = (P @ v).transpose(1, 2)

        return O, P, S

if __name__ == '__main__':
    dim = 384
    num_heads = 12
    head_size = 32
    window_size = 7
    window_num = 4

    q = torch.ones((49, 12, 32), dtype=torch.half).cuda()
    k = torch.ones((49, 12, 32), dtype=torch.half).cuda()
    v = torch.ones((49, 12, 32), dtype=torch.half).cuda()

    MHA = OriginMha(12)
    O, P, S =  MHA(q, k, v)

    print("q", q)
    print("k", k)
    print("v", v)
    print("P", P)
    print("S", S)
    print("O", O)
