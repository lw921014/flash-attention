import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from aiak_training.mlp import Linear as FusedLinear
from flash_attn.flash_attention import FlashMHA
from apex.normalization import FusedLayerNorm
from aiak_training.swin import WindowProcessFunc, WindowProcessReverseFunc
import numpy as np
import random as rand
import logging
import logging.handlers
from tools.check_tool import is_same_matrix

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None

logger = logging.getLogger("swin-T")
logger.setLevel(logging.INFO)

rf_handler = logging.StreamHandler()
rf_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(filename)s[:%(lineno)d] - %(message)s"))

logger.addHandler(rf_handler)

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., fused_linear=False, init_para = {}):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
       
        if fused_linear:
            self.qkv = FusedLinear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
       
        if fused_linear:
            self.proj = FusedLinear(dim, dim)
        else:
            self.proj = nn.Linear(dim, dim)

        self.proj_drop = nn.Dropout(proj_drop)

        if 'qkv_weight' in init_para.keys():
            self.qkv.weight = torch.nn.Parameter(init_para["qkv_weight"])
        if 'qkv_bias' in init_para.keys():
            self.qkv.bias = torch.nn.Parameter(init_para["qkv_bias"])
        if 'proj_weight' in init_para.keys():
            self.proj.weight = torch.nn.Parameter(init_para["proj_weight"])
        if 'proj_bias' in init_para.keys():
            self.proj.bias = torch.nn.Parameter(init_para["proj_bias"])

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

if __name__ == "__main__":
    dim = 192
    num_heads = 6
    window_size = 7
    qk_scale = None
    attn_drop = 0.0
    drop = 0.0
    fused_linear = True
    qkv_bias = True
    batch_size = 1

    x_windows = torch.rand(batch_size, window_size * window_size, dim, device='cuda', dtype=torch.half, requires_grad=True)
    resutlt = torch.rand(batch_size, window_size * window_size, dim, device='cuda', dtype=torch.half, requires_grad=False)

    loss = nn.L1Loss()

    wqkv_weight = np.random.uniform(-1, 1, [dim * 3, dim])
    wqkv_bias = np.random.uniform(-1, 1, [dim * 3])
    
    out_proj_weight = np.random.uniform(-1, 1, [dim, dim])
    out_proj_bias = np.random.uniform(-1, 1, [dim])

    init_para = {
        'qkv_weight' : torch.from_numpy(wqkv_weight),
        'qkv_bias' : torch.from_numpy(wqkv_bias),
        'proj_weight' : torch.from_numpy(out_proj_weight),
        'proj_bias' : torch.from_numpy(out_proj_bias)
    }

    attn_mask = None

    # win
    logger.info("windows attention")
    win_attn = WindowAttention(
        dim, window_size=to_2tuple(window_size), num_heads=num_heads,
        qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
        fused_linear=fused_linear, init_para = init_para).cuda()
    logger.info(f"win_attn model is \n {str(win_attn)}")

    print("para in win_attn is")
    for name, para in win_attn.named_parameters():
        print(name, para, para.shape)

    optimizer_win_attn = torch.optim.SGD(win_attn.parameters(), lr=1e-3)
    win_attn, optimizer_win_attn = amp.initialize(win_attn, optimizer_win_attn, opt_level="O2")

    win_output = win_attn(x_windows, attn_mask)
    win_loss = loss(win_output, resutlt)
    win_loss.backward()
    torch.cuda.synchronize()

    win_attn_grad = {}
    for name, parms in win_attn.named_parameters():	
        print('\nAfter backward\n')
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("===========================")

        if name == "qkv.weight":
            win_attn_grad["qkv_weight_grad"] = parms.grad.cpu().detach().numpy()
        if name == "qkv.bias":
            win_attn_grad["qkv_bias_grad"] = parms.grad.cpu().detach().numpy()
        if name == "proj.weight":
            win_attn_grad["proj_weight_grad"] = parms.grad.cpu().detach().numpy()
        if name == "proj.bias":
            win_attn_grad["proj_bias_grad"] = parms.grad.cpu().detach().numpy()

    # fmha
    logger.info("flash attention")
    flash_attn = FlashMHA(dim, to_2tuple(window_size), num_heads, init_para = init_para).cuda()
    logger.info(f"flash attn model is \n {str(flash_attn)}")

    print("para in flash_attn is")
    for name, para in flash_attn.named_parameters():
        print(name, para, para.shape)


    optimizer_win_fmha = torch.optim.SGD(flash_attn.parameters(), lr=1e-3)
    flash_attn, optimizer_win_fmha = amp.initialize(flash_attn, optimizer_win_fmha, opt_level="O2")
    flash_output = flash_attn(x_windows, None, None, attn_mask)
    flash_loss = loss(flash_output, resutlt)
    flash_loss.backward()
    torch.cuda.synchronize()

    flash_attn_grad = {}
    for name, parms in flash_attn.named_parameters():	
        print('\nAfter backward\n')
        print('-->name:', name)
        print('-->para:', parms)
        print('-->grad_requirs:',parms.requires_grad)
        print('-->grad_value:',parms.grad)
        print("===========================")

        if name == "Wqkv.weight":
            flash_attn_grad["qkv_weight_grad"] = parms.grad.cpu().detach().numpy()
        if name == "Wqkv.bias":
            flash_attn_grad["qkv_bias_grad"] = parms.grad.cpu().detach().numpy()
        if name == "out_proj.weight":
            flash_attn_grad["proj_weight_grad"] = parms.grad.cpu().detach().numpy()
        if name == "out_proj.bias":
            flash_attn_grad["proj_bias_grad"] = parms.grad.cpu().detach().numpy()

    # check output
    is_same_matrix(flash_output.cpu().detach().numpy(), win_output.cpu().detach().numpy(), "output")

    # check dgrad
    is_same_matrix(flash_attn_grad["qkv_weight_grad"], win_attn_grad["qkv_weight_grad"], "qkv_weight_grad")
    is_same_matrix(flash_attn_grad["qkv_bias_grad"], win_attn_grad["qkv_bias_grad"], "qkv_bias_grad")
    is_same_matrix(flash_attn_grad["proj_weight_grad"], win_attn_grad["proj_weight_grad"], "proj_weight_grad")
    is_same_matrix(flash_attn_grad["proj_bias_grad"], win_attn_grad["proj_bias_grad"], "proj_bias_grad")
