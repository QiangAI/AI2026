import torch
import torch.nn as nn
import torch.nn.quantized.dynamic as qdyn

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 256)
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

model = TextModel()
model.eval()

# 定义精确的类型映射
exact_mapping = {
    nn.LSTM: qdyn.LSTM,      # LSTM使用动态量化版本
    nn.Linear: qdyn.Linear,   # Linear使用动态量化版本
    # nn.Embedding: None       # Embedding不量化（因为没有动态版本）
}

# 使用mapping进行量化
quantized_model = torch.quantization.quantize_dynamic(
    model,
    mapping=exact_mapping,  # 使用精确类型映射
    inplace=False
)

# 验证替换结果
print("原始模型类型:")
for name, module in model.named_modules():
    print(f"  {name}: {type(module).__name__}")

print("\n量化后模型类型:")
for name, module in quantized_model.named_modules():
    print(f"  {name}: {type(module).__name__}")