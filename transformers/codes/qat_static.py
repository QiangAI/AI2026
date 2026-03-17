import warnings
warnings.filterwarnings("ignore")
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import time
import numpy as np
import os

# ======================1. 定义神经网络模型=================================
class QuantLeNet5(torch.nn.Module):
    def __init__(self, cls_num=10):
        super(QuantLeNet5, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = torch.nn.Conv2d(1, 6, 5, padding=2)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = torch.nn.Flatten(1)
        
        self.fc3   = torch.nn.Linear(16 * 5 * 5,  120)
        self.relu3 = torch.nn.ReLU(inplace=True)
        
        self.fc4   = torch.nn.Linear(120, 84)
        self.relu4 = torch.nn.ReLU(inplace=True)
        
        self.fc5   = torch.nn.Linear(84, cls_num)
        self.dequant = torch.ao.quantization.DeQuantStub()

    
    def forward(self, x):
        x = self.quant(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        
        x = self.fc3(x)
        x = self.relu3(x)
        
        x = self.fc4(x)
        x = self.relu4(x)

        x = self.fc5(x)
        
        x = self.dequant(x)
        return x

# ======================2. 加载训练与验证数据集=================================
def get_minst_laoder(root="./data", batch_size=128, num_calibration_batches=10):
    """
        num_calibration_batches用于校准的数据集批数。
    """
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.MNIST(root=root, train=True,  download=True, transform=transform)
    test_dataset  = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader =  torch.utils.data.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    # 校准数据集
    indices = torch.randperm(len(train_dataset))[:num_calibration_batches * batch_size]
    calibration_dataset = torch.utils.data.Subset(train_dataset, indices)
    calibration_loader = torch.utils.data.DataLoader(calibration_dataset, batch_size=batch_size, shuffle=False) 
    return train_loader, test_loader, calibration_loader
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
def train_model_minst(model, train_loader, test_loader, epochs=10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # ----------------------------------------------------------
    model.to(device)
    
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.ao.quantization.fuse_modules(
        model, 
        [
            ['conv1', 'relu1'],
            ['conv2', 'relu2'],
            ['fc3', 'relu3'],
            ['fc4', 'relu4'],
        ]
    )
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
            
            if batch_idx % 30 == 29:
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
    model_int8 = torch.ao.quantization.convert(model)
    return model_int8
    
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
model_fp32 = QuantLeNet5()
# 2. 加载数据集
print("加载数据集...")
train_loader, test_loader, _ = get_minst_laoder(batch_size=1000)
# 3. 量化感知训练
print("训练模型...")
model_fp32, losses, accs = train_model_minst(model_fp32, train_loader, test_loader, epochs=5)
# 4. 量化模型
print("量化模型...")
model_int8 = quantize_model(model_fp32)
# 5. 验证模型
print("验证非量化模型...")
accu1 = validate_model(model_fp32, test_loader, "cpu")
print(F"非量化模型准确度：{accu1 : 5.2f}%")
print("验证量化模型...")
accu2 = validate_model(model_int8, test_loader, "cpu")
print(F"量化模型准确度：{accu2 : 5.2f}%")

print("模型推理时间")
avg1, std1 = measure_inference_time(model_fp32, test_loader, "cpu")
avg2, std2 = measure_inference_time(model_int8, test_loader, "cpu")
print("量化前：", avg1, std1)
print("量化后：", avg2, std2)

print("模型大小")
size1 = get_model_size(model_fp32)
size2 = get_model_size(model_int8)
print("量化前：", size1)
print("量化后：", size2)