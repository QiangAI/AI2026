import copy
import torch
import torch.nn as nn
import torchvision
class ToyLinearModel(nn.Module):
    """适配CIFAR-10的ResNet18"""
    def __init__(self, num_classes=10):
        super().__init__()
        # 加载预训练的ResNet18
        self.model = torchvision.models.resnet152(weights=None)
        
        # 修改第一个卷积层以适应CIFAR-10的32x32输入
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 移除原始的池化层
        self.model.maxpool = nn.Identity()
        
        # 修改最后的全连接层
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

model = ToyLinearModel().cuda().to(torch.bfloat16).eval()

model_w16a16 = copy.deepcopy(model)
model_w8a8 = copy.deepcopy(model)  #

from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

quantize_(model_w8a8, Int8DynamicActivationInt8WeightConfig())
print(model_w8a8)

import os

# Save models
torch.save(model_w16a16.state_dict(), "model_w16a16.pth")
torch.save(model_w8a8.state_dict(), "model_w8a8.pth")

# Compare file sizes
original_size = os.path.getsize("model_w16a16.pth") / 1024**2
quantized_size = os.path.getsize("model_w8a8.pth") / 1024**2
print(
    f"Size reduction: {original_size / quantized_size:.2f}x ({original_size:.2f}MB -> {quantized_size:.2f}MB)"
)
