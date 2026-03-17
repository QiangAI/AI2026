import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import time
import numpy as np
import os
import copy
# ======================1. 定义神经网络模型=================================
class QuantVIT(torch.nn.Module):
    def __init__(self, original_model):
        super(QuantVIT, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()      # 用于把fp32转换为量化张量
        self.model = copy.deepcopy(original_model)  # 包装的模型（如果了解模型结构，可以选择性模块量化）
        self.dequant = torch.ao.quantization.DeQuantStub()  # 用于将输出的量化张量转换回 float32 张量
        
    def forward(self, inputs):
        x = self.quant(inputs)      # 将 float32 输入转换为量化张量
        y = self.model(x)    # 调用模型进行推理
        outputs = self.dequant(y)       # 将量化张量转换为fp32 （可以进一步处理）   
        return outputs

# ======================2. 加载训练与验证数据集=================================
def get_loader_imagenet(root="F:/04Datasets/ImageNet2012", batchsize=128):
    # 加载数据集
    _dataset_train = torchvision.datasets.ImageNet(
        root=root,
        split="val",
        transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms() # 需要是对象
    )
    
    _dataset_valid = torchvision.datasets.ImageNet(
        root=root,
        split="val",
        transform = torchvision.models.EfficientNet_B0_Weights.DEFAULT.transforms() # 需要是对象
    )
    
    num_calibration = 1000   # 总样本是50000 
    num_calibration = num_calibration if num_calibration<=len(_dataset_valid) else len(_dataset_valid)
    torch.manual_seed(42)
    indices = torch.randperm(len(_dataset_valid))[:num_calibration] + 1  # +1是因为randperm生成0-999
    _dataset_calib = torch.utils.data.Subset(_dataset_valid, indices)
    
    _loader_train = torch.utils.data.DataLoader(
        dataset=_dataset_train,        # 单样本数据集
        batch_size=batchsize,   # 数据集批次大小
        shuffle=True,  # 是否随机洗牌数据集 
    )
    
    _loader_valid = DataLoader(
        dataset=_dataset_valid,        # 单样本数据集
        batch_size=batchsize,   # 数据集批次大小
        shuffle=False,  # 是否随机洗牌数据集 
    )
    _loader_calib = DataLoader(
        dataset=_dataset_calib,        # 单样本数据集
        batch_size=batchsize,   # 数据集批次大小
        shuffle=False,  # 是否随机洗牌数据集 
    )

    return _loader_train, _loader_valid, _loader_calib

# ======================3. 模型验证===========================================
def validate_model(model, test_loader, device="cuda"):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_ = model(x)
            _, pred = y_.max(1)
            total += y.size(0)
            correct += (pred==y).sum().item()
    accuracy = 100. * correct / total
    return accuracy

# ======================4. 量化感知训练模型================================
def train_model_vit(model, train_loader, test_loader, epochs=10, device="cuda"):
    # ----------------------------------------------------------
    model.to(device)
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    # model = torch.ao.quantization.fuse_modules(
    #     model, 
    #     [
    #         ['conv1', 'relu1'],
    #         ['conv2', 'relu2'],
    #         ['fc3', 'relu3'],
    #         ['fc4', 'relu4'],
    #     ]
    # )
    model.train()
    model = torch.ao.quantization.prepare_qat(model)
    # ----------------------------------------------------------
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    
    train_losses = []
    test_accs = []
    
    for epoch in range(epochs):
        model.train()
        # 训练阶段
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_ = model(x)
            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, pred = y_.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            
            if batch_idx % 5 == 4:
                avg_loss = running_loss / 100
                accuracy = 100. * correct / total
                print(f'轮数: {epoch+1}, 批数: {batch_idx+1}, '
                      f'损失: {avg_loss:7.4f}, 精度: {accuracy:5.2f}%')
        
        # 调整学习率
        scheduler.step()
        # 验证
        test_acc = validate_model(model, test_loader)
        print(F"\t|- 验证精度：{test_acc:5.2f}%")
        test_accs.append(test_acc)
        train_losses.append(running_loss)
    torch.save(model.state_dict(), './models/tmp_model.pth')    
    return model, train_losses, test_accs

# ======================5. 对模型进行量化=================================
def quantize_model(model):
    model.to("cpu")
    model.eval()
    _model_int8 = torch.ao.quantization.convert(model)
    return _model_int8

# ======================6. 评估推理时间=================================
def measure_inference_time(model, test_loader, device='cpu', num_iterations=100):
    model.eval()
    model.to(device)
    # 预热
    for data, _ in test_loader:
        data = data.to(device)
        _ = model(data)
        break
    # 测量时间
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= num_iterations:
                break
            data = data.to(device)
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            _ = model(data)
            if device == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # 转换为毫秒
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time
# ======================7. 获取模型大小=================================
def get_model_size(model):
    torch.save(model.state_dict(), 'temp_model.pth')
    size_mb = os.path.getsize('temp_model.pth') / (1024 * 1024)
    os.remove('temp_model.pth')
    return size_mb

# ======================8. 执行=================================
# 1. 创建与初始化模型
print("创建模型...")
model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
model_fp32 = QuantVIT(model)

# 2. 加载数据集
print("加载数据集...")
train_loader, test_loader, calib_loader = get_loader_imagenet(batchsize=100)

# 3. 量化感知训练
print("训练模型...")
model_fp32, losses, accs = train_model_vit(model_fp32, calib_loader, calib_loader, epochs=2)

# 4. 量化模型
print("量化模型...")
model_int8 = quantize_model(model_fp32)

# 5. 验证模型
print("验证非量化模型...")
accu1 = validate_model(model_fp32, calib_loader, "cpu")
print(F"非量化模型准确度：{accu1 : 5.2f}%")
print("验证量化模型...")
accu2 = validate_model(model_int8, calib_loader, "cpu")
print(F"量化模型准确度：{accu2 : 5.2f}%")

print("模型推理时间")
avg1, std1 = measure_inference_time(model_fp32, calib_loader, "cpu")
avg2, std2 = measure_inference_time(model_int8, calib_loader, "cpu")
print("量化前：", avg1, std1)
print("量化后：", avg2, std2)

print("模型大小")
size1 = get_model_size(model_fp32)
size2 = get_model_size(model_int8)
print("量化前：", size1)
print("量化后：", size2)