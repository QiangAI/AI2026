import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
import copy
plt.rcParams["font.family"] = ["Microsoft YaHei"]
torch.manual_seed(42)
np.random.seed(42)

data_dir = "F:/04Datasets/CIFAR10"
# 1. 数据集 - CIFAR10(10个类别：'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
def prepare_data(batch_size=128, subset_size=None):
    """准备CIFAR-10数据集"""
    # 数据预处理
    transform_train = transforms.Compose([   # 训练数据集预处理（数据增强：裁剪与翻转）
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([  # 测试数据集预处理
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 加载CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    # 可选：使用子集加速训练
    if subset_size:
        indices = torch.randperm(len(trainset))[:subset_size]
        trainset = Subset(trainset, indices)
        
        indices = torch.randperm(len(testset))[:subset_size//5]
        testset = Subset(testset, indices)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader

# 加载数据
trainloader, testloader = prepare_data(batch_size=64, subset_size=10000)  # 使用子集加速演示
# print(f"训练集大小: {len(trainloader.dataset)}")
# print(f"测试集大小: {len(testloader.dataset)}")
# for x, y in trainloader:
#     print("输入格式：", x.shape)
#     print("标签格式", y.shape)
#     break

# 2. 模型定义
class ResNetForCIFAR(nn.Module):
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

# model = ResNetForCIFAR()
# for x, y in trainloader:
#     # y_ = model.forward(x)
#     y_ = model(x)
#     print(y_.shape)
#     break

def train_one_epoch(model, trainloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    # model = torch.compile(model)
    pbar = tqdm(trainloader, desc='训练')
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        pbar.set_postfix({'Loss': f'{running_loss/total:.3f}', 
                         'Acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(trainloader), 100. * correct / total

def evaluate(model, testloader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc='|-评估'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1) # _返回最大概率（需要使用softmax或者sigmoid处理），predicted返回预测类别（最大概率下标索引）
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(testloader), 100. * correct / total

def count_parameters(model):
    """统计模型参数"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_time(model, testloader, device, num_batches=50):
    """测量推理时间"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(testloader):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            
            # 预热
            if i == 0:
                for _ in range(10):
                    _ = model(inputs)
            
            # 测量时间
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            _ = model(inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            
            times.append((end - start) * 1000)  # 转换为毫秒
    
    return np.mean(times), np.std(times)

def get_model_size(model):
    """估算模型内存大小（MB）"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = (param_size + buffer_size) / 1024**2
    return total_size

from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_
from torchao.quantization import Int8WeightOnlyConfig
from torchao.sparsity import sparsify_, semi_sparse_weight

def apply_quantization(model, quant_type='int8_weight_only', example_inputs=None):
    """
    应用不同类型的量化
    
    Args:
        model: 要量化的模型
        quant_type: 量化类型
        example_inputs: 用于校准的示例输入
    """
    if quant_type == 'int8_weight_only':
        # 仅权重INT8量化
        quantize_(model, Int8WeightOnlyConfig(version=2))
        
    elif quant_type == 'int8_dynamic':
        # 动态INT8量化（激活也量化）
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        
    elif quant_type == 'int8_sparse':
        # 半稀疏INT8量化
        sparsify_(model, semi_sparse_weight()) 
    return model

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    print(f"使用设备: {device}")
    
    # 创建基线模型
    print("\n创建基线模型...")
    baseline_model = ResNetForCIFAR().to(device).to(dtype)
    print(f"基线模型参数数量: {count_parameters(baseline_model):,}")
    print(f"基线模型大小: {get_model_size(baseline_model):.2f} MB")
    
    # 训练基线模型
    print("\n" + "=" * 100)
    print("训练基线模型")
    print("=" * 100)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(baseline_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    baseline_history = {'train_acc': [], 'test_acc': []}
    
    for epoch in range(3):  # 减少epoch以加快演示
        train_loss, train_acc = train_one_epoch(baseline_model, trainloader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(baseline_model, testloader, criterion, device)
        scheduler.step()
        
        baseline_history['train_acc'].append(train_acc)
        baseline_history['test_acc'].append(test_acc)
        
        print(f'Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    
    # 评估不同量化方法
    print("\n" + "="*60)
    print("评估不同量化方法")
    print("="*60)
    
    quant_methods = [
        ('int8_weight_only', 'INT8权重量化'),
        ('int8_dynamic', '动态INT8量化'),
        # ('int8_sparse', "稀疏INT8量化")
    ]
    
    results = {
        'baseline': {
            'accuracy': baseline_history['test_acc'][-1],
            'size': get_model_size(baseline_model),
            'inference_time': measure_inference_time(baseline_model, testloader, device)
        }
    }
    
    for quant_type, quant_name in quant_methods:
        print(f"\n测试: {quant_name}")
        
        # 复制模型并应用量化
        quant_model = copy.deepcopy(baseline_model)
        quant_model = apply_quantization(quant_model, quant_type)
        
        # 评估
        test_loss, test_acc = evaluate(quant_model, testloader, criterion, device)
        model_size = get_model_size(quant_model)
        inference_time, std_time = measure_inference_time(quant_model, testloader, device)
        
        results[quant_type] = {
            'accuracy': test_acc,
            'size': model_size,
            'inference_time': (inference_time, std_time)
        }
        
        print(f"准确率: {test_acc:.2f}%")
        print(f"模型大小: {model_size:.2f} MB")
        print(f"推理时间: {inference_time:.2f} ± {std_time:.2f} ms")
    
    # 可视化结果
    visualize_results(results)
    
    return results

def visualize_results(results):
    """可视化比较结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = list(results.keys())
    method_names = ['Baseline', 'INT8\nWeight', 'INT8\nDynamic']
    
    # 准确率比较
    accuracies = [results[m]['accuracy'] for m in methods]
    axes[0].bar(method_names, accuracies, color=['blue', 'orange', 'green', 'red', 'purple'])
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('模型准确率比较')
    axes[0].set_ylim([0, 100])
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # 模型大小比较
    sizes = [results[m]['size'] for m in methods]
    axes[1].bar(method_names, sizes, color=['blue', 'orange', 'green', 'red', 'purple'])
    axes[1].set_ylabel('Model Size (MB)')
    axes[1].set_title('模型大小比较')
    for i, v in enumerate(sizes):
        axes[1].text(i, v + 0.1, f'{v:.1f}MB', ha='center')
    
    # 推理时间比较
    times = [results[m]['inference_time'][0] for m in methods]
    errors = [results[m]['inference_time'][1] for m in methods]
    axes[2].bar(method_names, times, yerr=errors, capsize=5,
                color=['blue', 'orange', 'green', 'red', 'purple'])
    axes[2].set_ylabel('Inference Time (ms)')
    axes[2].set_title('推理时间比较')
    
    plt.tight_layout()
    plt.savefig('quantization_comparison.png', dpi=150)
    plt.show()
    print("\n结果图已保存为 'quantization_comparison.png'")

if __name__ == "__main__":
    # 运行主程序
    results = main()
