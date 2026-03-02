import torch
import torch.nn as nn

from torchao.quantization import Int8WeightOnlyConfig, quantize_, PerRow

# create model and sample input
m = nn.Sequential(
        nn.Linear(8192, 4096, bias=False),
        nn.Linear(4096, 128, bias=False),
    ).float().cuda()

optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

# configure float8 recipe
# valid recipe names: "tensorwise", "rowwise", "rowwise_with_gw_hp"
config = Int8WeightOnlyConfig(granularity=PerRow())

# convert specified `torch.nn.Linear` modules to `Float8Linear`
quantize_(m, config=config)
# enable torch.compile for competitive performance
# m = torch.compile(m, mode="default")

# training loop
x = torch.randn(4096, 8192, device="cuda", dtype=torch.float32)
for _ in range(10):
    optimizer.zero_grad()
    y = m(x)
    loss = y.sum()
    loss.backward()
    optimizer.step()
