import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from flash_attn.flash_attention import FlashMHA
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

if __name__ == '__main__':
    dim = 384
    num_heads = 12
    head_size = 32
    window_size = 7
    window_num = 4
    attn_flash = FlashMHA(dim, to_2tuple(window_size), num_heads, dtype=torch.half).cuda()

    x_windows = torch.ones((128, window_size * window_size, dim), dtype=torch.half).cuda()
    attn_mask = torch.ones((window_num, window_size * window_size, window_size * window_size), dtype=torch.half).cuda()
    result = torch.rand((128, window_size * window_size, dim), dtype=torch.half).cuda()

    # attn_windows = attn_flash(x_windows, None, None, attn_mask=attn_mask)

    # print(attn_windows.size())
    optimzer = torch.optim.SGD(attn_flash.parameters(), lr=0.05) 
    loss_func = nn.MSELoss() 
    for epoch in range(10):                   
        out = attn_flash(x_windows, None, None, attn_mask=attn_mask)                         
        loss = loss_func(out, result) # 计算误差         
        optimzer.zero_grad()  # 清除上一次的梯度        
        loss.backward()    # 让误差反向传播            
        optimzer.step()     # 让优化器工作
