# import os
# os.environ['TORCH_SCALED_MM'] = "0"
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F

import torch
import torch.nn as nn
from torchao import quantize_
from torchao.quantization.quant_api import Int8WeightOnlyConfig


# 模型
m = nn.Sequential(
    nn.Linear(2048, 4096),
    nn.Linear(4096, 128),
    nn.Linear(128, 1),
).bfloat16().cuda()

m2 = copy.deepcopy(m)
# 输入数据
x = torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16)
# 优化器
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
# 损失函数直接使用均方差损失mse_loss


# 转换`torch.nn.Linear`模块到`Float8Linear`
quantize_(m2, Int8WeightOnlyConfig(group_size=32))
# 启用torch.compile以提升性能
m = torch.compile(m)
m2 = torch.compile(m2)
# toy training loop
start = time.time()
for _ in range(10):
    optimizer.zero_grad()
    output = m(x)
    # # 模拟标签
    fake_labels = torch.ones_like(output)  
    loss = F.mse_loss(output, fake_labels)
    loss.backward()
    optimizer.step()
custom_time = time.time() - start
print("普通模型运行时间：", custom_time)

start = time.time()
for _ in range(10):
    optimizer.zero_grad()
    output = m2(x)
    # # 模拟标签
    fake_labels = torch.ones_like(output)  
    loss = F.mse_loss(output, fake_labels)
    loss.backward()
    optimizer.step()
custom_time = time.time() - start
print("量化模型运行时间", custom_time)
# torch.save(m.state_dict(), './outs/checkpoint8.pth')
