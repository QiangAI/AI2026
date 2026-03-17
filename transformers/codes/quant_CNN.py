"""
Eager模式训练后静态量化完整示例
包含：模型定义、训练、量化、评估和部署
"""
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub, prepare, convert
from torch.quantization import get_default_qconfig
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
import os
import copy
import matplotlib.pyplot as plt
from collections import OrderedDict

plt.rcParams["font.family"] = ["Microsoft YaHei"]

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 1. 模型定义 ====================
class QuantizableCNN(nn.Module):
    """
    支持量化的CNN模型
    包含QuantStub和DeQuantStub用于静态量化
    """
    def __init__(self, num_classes=10):
        super(QuantizableCNN, self).__init__()
        
        # 量化存根 - 必须添加
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # 特征提取层
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, kernel_size=3, padding=1)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            
            ('conv2', nn.Conv2d(32, 64, kernel_size=3, padding=1)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2)),
            
            ('conv3', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('relu3', nn.ReLU()),
            ('pool3', nn.MaxPool2d(2)),
        ]))
        
        # 分类器
        self.classifier = nn.Sequential(OrderedDict([
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(128 * 3 * 3, 256)),
            ('relu4', nn.ReLU()),
            ('dropout', nn.Dropout(0.3)),
            ('fc2', nn.Linear(256, num_classes))
        ]))
        
        # 合并相邻的模块以提高量化效率
        self._merge_modules()
    
    def _merge_modules(self):
        """合并相邻的ReLU和卷积层以提高量化效率"""
        # 在实际应用中，可以使用torch.quantization.fuse_modules
        # 这里为了演示，我们保持原样
        pass
    
    def forward(self, x):
        # 量化输入
        x = self.quant(x)
        
        # 特征提取
        x = self.features(x)
        
        # 分类
        x = self.classifier(x)
        
        # 反量化输出
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """
        融合模型中的模块以提高量化效率
        将Conv+ReLU或Linear+ReLU融合在一起
        """
        # 融合特征提取层中的Conv+ReLU
        torch.quantization.fuse_modules(
            self.features,
            [['conv1', 'relu1'],
             ['conv2', 'relu2'],
             ['conv3', 'relu3']],
            inplace=True
        )
        
        # 融合分类器中的Linear+ReLU
        torch.quantization.fuse_modules(
            self.classifier,
            [['fc1', 'relu4']],
            inplace=True
        )
        
        print("模型模块融合完成")

# ==================== 2. 数据准备 ====================
class DataPreparer:
    """数据准备类"""
    
    @staticmethod
    def get_mnist_loaders(batch_size=128, num_calibration_batches=10):
        """
        准备MNIST数据集加载器
        
        Args:
            batch_size: 批次大小
            num_calibration_batches: 用于校准的batch数量
            
        Returns:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            calibration_loader: 校准数据加载器
        """
        
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 加载数据集
        train_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=True, 
            download=True, 
            transform=transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root='./data', 
            train=False, 
            download=True, 
            transform=transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # 创建用于校准的数据加载器（使用训练数据的一小部分）
        indices = torch.randperm(len(train_dataset))[:num_calibration_batches * batch_size]
        calibration_dataset = Subset(train_dataset, indices)
        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"训练集大小: {len(train_dataset)}")
        print(f"测试集大小: {len(test_dataset)}")
        print(f"校准集大小: {len(calibration_dataset)}")
        
        return train_loader, test_loader, calibration_loader

# ==================== 3. 训练函数 ====================
class ModelTrainer:
    """模型训练类"""
    
    @staticmethod
    def train(model, train_loader, test_loader, epochs=5, device='cpu'):
        """
        训练模型
        
        Args:
            model: 待训练的模型
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            epochs: 训练轮数
            device: 训练设备
        """
        
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        
        best_acc = 0.0
        train_losses = []
        test_accs = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                if batch_idx % 100 == 99:
                    avg_loss = running_loss / 100
                    accuracy = 100. * correct / total
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, '
                          f'Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
                    running_loss = 0.0
            
            # 调整学习率
            scheduler.step()
            
            # 测试阶段
            test_acc = ModelEvaluator.evaluate(model, test_loader, device)
            test_accs.append(test_acc)
            train_losses.append(running_loss)
            
            print(f'Epoch {epoch+1} 完成，测试准确率: {test_acc:.2f}%')
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), './models/best_model.pth')
                print(f'保存最佳模型，准确率: {best_acc:.2f}%')
        
        # 加载最佳模型
        model.load_state_dict(torch.load('./models/best_model.pth'))
        
        return model, train_losses, test_accs

# ==================== 4. 模型评估 ====================
class ModelEvaluator:
    """模型评估类"""
    
    @staticmethod
    def evaluate(model, test_loader, device='cpu'):
        """
        评估模型准确率
        
        Returns:
            accuracy: 准确率百分比
        """
        model.eval()
        model.to(device)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    @staticmethod
    def measure_inference_time(model, test_loader, device='cpu', num_iterations=100):
        """
        测量推理时间
        
        Returns:
            avg_time: 平均推理时间（毫秒）
            std_time: 标准差
        """
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
    
    @staticmethod
    def get_model_size(model):
        """
        获取模型大小
        
        Returns:
            size_mb: 模型大小（MB）
        """
        torch.save(model.state_dict(), 'temp_model.pth')
        size_mb = os.path.getsize('temp_model.pth') / (1024 * 1024)
        os.remove('temp_model.pth')
        return size_mb

# ==================== 5. 量化函数 ====================
class PostTrainingQuantizer:
    """
    训练后静态量化类
    """
    
    def __init__(self, backend='fbgemm'):
        """
        初始化量化器
        
        Args:
            backend: 量化后端 ('fbgemm' 或 'qnnpack')
        """
        self.backend = backend
        self.supported_backends = ['fbgemm', 'qnnpack']
        
        if backend not in self.supported_backends:
            raise ValueError(f"不支持的量化后端: {backend}")
        
        print(f"使用量化后端: {backend}")
    
    def prepare_model_for_quantization(self, model):
        """
        准备模型进行量化
        
        Steps:
        1. 融合模块
        2. 设置量化配置
        3. 准备模型
        """
        
        # 1. 创建模型副本
        model_quant = copy.deepcopy(model)
        model_quant.eval()
        
        # 2. 融合模块（可选，但强烈推荐）
        try:
            model_quant.fuse_model()
        except AttributeError:
            print("模型不支持自动融合，跳过融合步骤")
        
        # 3. 设置量化配置
        model_quant.qconfig = get_default_qconfig(self.backend)
        print(f"量化配置: {model_quant.qconfig}")
        
        # 4. 准备模型
        model_prepared = prepare(model_quant, inplace=False)
        print("模型准备完成，已插入观察者")
        
        return model_prepared
    
    def calibrate(self, model_prepared, calibration_loader):
        """
        校准模型
        
        Args:
            model_prepared: 准备好的模型
            calibration_loader: 校准数据加载器
        """
        print("开始校准...")
        model_prepared.eval()
        
        with torch.no_grad():
            for i, (data, _) in enumerate(calibration_loader):
                model_prepared(data)
                
                # 可选：显示校准进度
                if (i + 1) % 10 == 0:
                    print(f"校准进度: {i + 1}/{len(calibration_loader)}")
        
        print("校准完成")
        return model_prepared
    
    def convert_model(self, model_prepared):
        """
        转换模型为量化版本
        
        Returns:
            model_quantized: 量化后的模型
        """
        print("开始转换模型...")
        model_quantized = convert(model_prepared, inplace=False)
        print("模型转换完成")
        
        return model_quantized
    
    def quantize(self, model, calibration_loader):
        """
        完整的量化流程
        
        Args:
            model: 原始模型
            calibration_loader: 校准数据加载器
            
        Returns:
            model_quantized: 量化后的模型
        """
        # 准备
        model_prepared = self.prepare_model_for_quantization(model)
        
        # 校准
        model_calibrated = self.calibrate(model_prepared, calibration_loader)
        
        # 转换
        model_quantized = self.convert_model(model_calibrated)
        
        return model_quantized

# ==================== 6. 实验结果可视化 ====================
class ResultVisualizer:
    """结果可视化类"""
    
    @staticmethod
    def plot_comparison(fp32_acc, quantized_acc, fp32_time, quantized_time, 
                        fp32_size, quantized_size):
        """
        绘制对比图表
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 准确率对比
        axes[0].bar(['FP32', 'INT8'], [fp32_acc, quantized_acc], 
                    color=['blue', 'orange'])
        axes[0].set_ylabel('准确率 (%)')
        axes[0].set_title('模型准确率对比')
        axes[0].set_ylim([95, 100])
        for i, v in enumerate([fp32_acc, quantized_acc]):
            axes[0].text(i, v + 0.1, f'{v:.2f}%', ha='center')
        
        # 推理时间对比
        axes[1].bar(['FP32', 'INT8'], [fp32_time, quantized_time],
                    color=['blue', 'orange'])
        axes[1].set_ylabel('推理时间 (ms)')
        axes[1].set_title('推理时间对比')
        for i, v in enumerate([fp32_time, quantized_time]):
            axes[1].text(i, v + 0.1, f'{v:.2f}ms', ha='center')
        
        # 模型大小对比
        axes[2].bar(['FP32', 'INT8'], [fp32_size, quantized_size],
                    color=['blue', 'orange'])
        axes[2].set_ylabel('模型大小 (MB)')
        axes[2].set_title('模型大小对比')
        for i, v in enumerate([fp32_size, quantized_size]):
            axes[2].text(i, v + 0.01, f'{v:.2f}MB', ha='center')
        
        plt.tight_layout()
        plt.savefig('quantization_comparison.png')
        plt.show()
        print("对比图表已保存为 'quantization_comparison.png'")
    
    @staticmethod
    def print_summary_table(fp32_metrics, quantized_metrics):
        """
        打印总结表格
        """
        print("\n" + "="*60)
        print("模型量化效果总结")
        print("="*60)
        print(f"{'指标':<20} {'FP32模型':<15} {'INT8模型':<15} {'变化':<15}")
        print("-"*60)
        
        metrics = [
            ('准确率 (%)', f"{fp32_metrics['accuracy']:.2f}", 
             f"{quantized_metrics['accuracy']:.2f}", 
             f"{quantized_metrics['accuracy'] - fp32_metrics['accuracy']:.2f}"),
            ('推理时间 (ms)', f"{fp32_metrics['time']:.2f}", 
             f"{quantized_metrics['time']:.2f}", 
             f"{(fp32_metrics['time']/quantized_metrics['time']):.2f}x 加速"),
            ('模型大小 (MB)', f"{fp32_metrics['size']:.2f}", 
             f"{quantized_metrics['size']:.2f}", 
             f"{(fp32_metrics['size']/quantized_metrics['size']):.2f}x 压缩")
        ]
        
        for name, fp32_val, quant_val, change in metrics:
            print(f"{name:<20} {fp32_val:<15} {quant_val:<15} {change:<15}")
        
        print("="*60)

# ==================== 7. 主函数 ====================
def main():
    """主函数：执行完整的训练后静态量化流程"""
    
    print("\n" + "="*60)
    print("训练后静态量化完整示例")
    print("="*60)
    
    # 创建必要的目录
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    
    # 1. 准备数据
    print("\n1. 准备数据...")
    data_preparer = DataPreparer()
    train_loader, test_loader, calibration_loader = data_preparer.get_mnist_loaders(
        batch_size=128,
        num_calibration_batches=20
    )
    
    # 2. 创建或加载模型
    print("\n2. 创建模型...")
    model = QuantizableCNN(num_classes=10)
    
    # 检查是否有预训练模型
    model_path = './models/best_model.pth'
    if os.path.exists(model_path):
        print("加载预训练模型...")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("训练新模型...")
        trainer = ModelTrainer()
        model, _, _ = trainer.train(
            model, train_loader, test_loader, 
            epochs=3, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    # 3. 评估原始FP32模型
    print("\n3. 评估原始FP32模型...")
    fp32_accuracy = ModelEvaluator.evaluate(model, test_loader, device='cpu')
    fp32_time, fp32_std = ModelEvaluator.measure_inference_time(
        model, test_loader, device='cpu'
    )
    fp32_size = ModelEvaluator.get_model_size(model)
    
    print(f"FP32模型准确率: {fp32_accuracy:.2f}%")
    print(f"FP32模型推理时间: {fp32_time:.2f} ± {fp32_std:.2f} ms")
    print(f"FP32模型大小: {fp32_size:.2f} MB")
    
    # 4. 执行训练后静态量化
    print("\n4. 执行训练后静态量化...")
    
    # 创建量化器（使用fbgemm后端）
    quantizer = PostTrainingQuantizer(backend='fbgemm')
    
    # 执行量化
    model_quantized = quantizer.quantize(model, calibration_loader)
    print("量化后参数")
    print(list(model_quantized.parameters()))
    # 5. 评估量化模型
    print("\n5. 评估量化模型...")
    quantized_accuracy = ModelEvaluator.evaluate(model_quantized, test_loader, device='cpu')
    quantized_time, quantized_std = ModelEvaluator.measure_inference_time(
        model_quantized, test_loader, device='cpu'
    )
    quantized_size = ModelEvaluator.get_model_size(model_quantized)
    
    print(f"INT8模型准确率: {quantized_accuracy:.2f}%")
    print(f"INT8模型推理时间: {quantized_time:.2f} ± {quantized_std:.2f} ms")
    print(f"INT8模型大小: {quantized_size:.2f} MB")
    
    # 6. 保存模型
    print("\n6. 保存模型...")
    torch.save(model_quantized.state_dict(), './models/quantized_model.pth')
    print("量化模型已保存到 './models/quantized_model.pth'")
    
    # 7. 导出为TorchScript（用于部署）
    print("\n7. 导出为TorchScript...")
    try:
        scripted_model = torch.jit.script(model_quantized)
        scripted_model.save('./models/quantized_model_scripted.pt')
        print("TorchScript模型已保存")
    except Exception as e:
        print(f"TorchScript导出失败: {e}")
    
    # 8. 结果汇总
    print("\n8. 结果汇总...")
    fp32_metrics = {
        'accuracy': fp32_accuracy,
        'time': fp32_time,
        'size': fp32_size
    }
    
    quantized_metrics = {
        'accuracy': quantized_accuracy,
        'time': quantized_time,
        'size': quantized_size
    }
    
    # 打印总结表格
    ResultVisualizer.print_summary_table(fp32_metrics, quantized_metrics)
    
    # 绘制对比图表
    ResultVisualizer.plot_comparison(
        fp32_accuracy, quantized_accuracy,
        fp32_time, quantized_time,
        fp32_size, quantized_size
    )
    
    print("\n训练后静态量化完成！")

# ==================== 8. 高级量化配置示例 ====================
def advanced_quantization_examples():
    """
    高级量化配置示例
    展示不同的量化选项和配置
    """
    
    print("\n" + "="*60)
    print("高级量化配置示例")
    print("="*60)
    
    # 1. 不同的量化后端
    print("\n1. 不同的量化后端:")
    backends = ['fbgemm', 'qnnpack']
    for backend in backends:
        if backend in torch.backends.quantized.supported_engines:
            qconfig = get_default_qconfig(backend)
            print(f"   {backend}: {qconfig}")
    
    # 2. 自定义量化配置
    print("\n2. 自定义量化配置:")
    
    # 使用不同的观察者
    custom_qconfigs = {
        'minmax': torch.quantization.QConfig(
            activation=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            ),
            weight=torch.quantization.MinMaxObserver.with_args(
                dtype=torch.qint8,
                qscheme=torch.per_tensor_symmetric
            )
        ),
        'histogram': torch.quantization.QConfig(
            activation=torch.quantization.HistogramObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            ),
            weight=torch.quantization.default_weight_observer
        )
    }
    
    for name, qconfig in custom_qconfigs.items():
        print(f"   {name}: {qconfig}")
    
    # 3. 逐层量化配置
    print("\n3. 逐层量化配置示例:")
    
    model = QuantizableCNN()
    
    # 为不同层设置不同的量化配置
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 卷积层使用更精确的量化
            module.qconfig = torch.quantization.QConfig(
                activation=torch.quantization.HistogramObserver.with_args(
                    dtype=torch.quint8
                ),
                weight=torch.quantization.default_weight_observer
            )
            print(f"   设置 {name} 使用直方图量化")
        
        elif isinstance(module, nn.Linear):
            # 全连接层使用较快的量化
            module.qconfig = get_default_qconfig('fbgemm')
            print(f"   设置 {name} 使用默认量化")
    
    # 4. 模型融合示例
    print("\n4. 模型融合示例:")
    print("   融合前模块数:", sum(1 for _ in model.modules()))
    
    # 手动指定融合层
    try:
        torch.quantization.fuse_modules(
            model.features,
            [['conv1', 'relu1'],
             ['conv2', 'relu2'],
             ['conv3', 'relu3']],
            inplace=True
        )
        print("   融合后模块数:", sum(1 for _ in model.modules()))
    except Exception as e:
        print(f"   融合失败: {e}")

# ==================== 9. 部署示例 ====================
def deployment_example():
    """
    部署示例：展示如何在推理中使用量化模型
    """
    
    print("\n" + "="*60)
    print("量化模型部署示例")
    print("="*60)
    
    # 1. 加载量化模型
    print("\n1. 加载量化模型...")
    
    # 方法1：直接加载state_dict
    model = QuantizableCNN()
    model.eval()
    
    # 设置量化配置
    model.qconfig = get_default_qconfig('fbgemm')
    
    # 准备模型（需要和量化时相同的配置）
    model_prepared = prepare(model, inplace=False)
    
    # 转换为量化版本
    model_quantized = convert(model_prepared, inplace=False)
    
    # 加载量化权重
    if os.path.exists('./models/quantized_model.pth'):
        model_quantized.load_state_dict(torch.load('./models/quantized_model.pth'))
        print("   量化模型加载成功")
    else:
        print("   未找到量化模型文件")
        return
    
    # 方法2：加载TorchScript模型
    if os.path.exists('./models/quantized_model_scripted.pt'):
        scripted_model = torch.jit.load('./models/quantized_model_scripted.pt')
        print("   TorchScript模型加载成功")
    
    # 2. 推理示例
    print("\n2. 推理示例...")
    
    # 创建示例输入
    example_input = torch.randn(1, 1, 28, 28)
    
    # 执行推理
    with torch.no_grad():
        output = model_quantized(example_input)
        probabilities = F.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
    
    print(f"   输入形状: {example_input.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   预测类别: {predicted_class.item()}")
    print(f"   类别概率: {probabilities[0][predicted_class].item():.4f}")
    
    # 3. 性能测试
    print("\n3. 部署性能测试...")
    
    # 批量推理
    batch_sizes = [1, 4, 8, 16, 32]
    print(f"{'批次大小':<10} {'推理时间(ms)':<15} {'吞吐量(样本/s)':<15}")
    print("-" * 40)
    
    for batch_size in batch_sizes:
        batch_input = torch.randn(batch_size, 1, 28, 28)
        
        # 预热
        for _ in range(5):
            _ = model_quantized(batch_input)
        
        # 测量时间
        times = []
        for _ in range(50):
            start = time.time()
            _ = model_quantized(batch_input)
            end = time.time()
            times.append((end - start) * 1000)
        
        avg_time = np.mean(times)
        throughput = (batch_size / avg_time) * 1000  # 样本/秒
        
        print(f"{batch_size:<10} {avg_time:<15.2f} {throughput:<15.2f}")

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    # 运行主流程
    main()
    
    # 展示高级配置
    advanced_quantization_examples()
    
    # 部署示例
    deployment_example()