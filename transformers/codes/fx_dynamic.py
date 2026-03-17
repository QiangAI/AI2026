import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 1. 定义一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# 2. 创建一个带有更多层和不同操作的复杂模型
class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 16 * 16, 128)  # 假设输入为32x32
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 3. 动态量化的主要函数
def dynamic_quantization_example(model, model_name="SimpleModel"):
    print(f"\n{'='*50}")
    print(f"动态量化示例 - {model_name}")
    print(f"{'='*50}")
    
    # 创建模型实例
    model.eval()
    
    # 打印原始模型结构
    print(f"\n原始模型结构:")
    print(model)
    
    # 准备输入数据
    if isinstance(model, SimpleModel):
        sample_input = torch.randn(1, 1, 28, 28)
        # 对于SimpleModel，我们需要reshape为(1, 784)
        flattened_input = sample_input.view(1, -1)
    else:  # ComplexModel
        sample_input = torch.randn(1, 3, 32, 32)
        flattened_input = sample_input
    
    # 获取原始模型的输出
    with torch.no_grad():
        original_output = model(flattened_input)
    
    print(f"\n原始模型输出 (前5个值): {original_output[0][:5]}")
    print(f"原始模型输出数据类型: {original_output.dtype}")
    
    # 4. 配置动态量化
    # 对于动态量化，我们通常只量化线性层和LSTM层
    qconfig_dict = {
        "": torch.ao.quantization.default_dynamic_qconfig,
    }
    
    # 或者更精确地指定哪些模块需要量化
    # qconfig_dict = {
    #     "fc1": torch.ao.quantization.default_dynamic_qconfig,
    #     "fc2": torch.ao.quantization.default_dynamic_qconfig,
    #     "fc3": torch.ao.quantization.default_dynamic_qconfig,
    # }
    
    # 5. 准备动态量化
    print(f"\n准备动态量化...")
    model_prepared = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs=(flattened_input,))
    
    # 6. 转换为量化模型
    print(f"转换为量化模型...")
    model_quantized = quantize_fx.convert_fx(model_prepared)
    
    # 打印量化后的模型结构
    print(f"\n量化后的模型结构:")
    print(model_quantized)
    
    # 7. 测试量化模型
    with torch.no_grad():
        quantized_output = model_quantized(flattened_input)
    
    print(f"\n量化模型输出 (前5个值): {quantized_output[0][:5]}")
    print(f"量化模型输出数据类型: {quantized_output.dtype}")
    
    # 8. 比较输出差异
    diff = torch.abs(original_output - quantized_output)
    print(f"\n输出差异统计:")
    print(f"  最大差异: {diff.max().item():.6f}")
    print(f"  平均差异: {diff.mean().item():.6f}")
    print(f"  中位数差异: {diff.median().item():.6f}")
    
    return model_quantized

# 4. 自定义动态量化配置的示例
def custom_dynamic_quantization_example():
    print(f"\n{'='*50}")
    print(f"自定义动态量化配置示例")
    print(f"{'='*50}")
    
    # 创建一个简单的模型
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 5)
            self.fc3 = nn.Linear(5, 2)
            
        def forward(self, x):
            x = self.fc1(x)
            x = torch.sigmoid(x)
            x = self.fc2(x)
            x = torch.relu(x)
            x = self.fc3(x)
            return x
    
    model = CustomModel().eval()
    sample_input = torch.randn(1, 10)
    
    # 自定义量化配置 - 使用不同的量化参数
    from torch.ao.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver
    
    custom_qconfig = torch.ao.quantization.QConfig(
        activation=MinMaxObserver.with_args(dtype=torch.quint8, qscheme=torch.per_tensor_affine),
        weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_symmetric)
    )
    
    # 为不同层应用不同的量化配置
    qconfig_dict = {
        "": custom_qconfig,  # 默认配置
        "fc1": torch.ao.quantization.default_dynamic_qconfig,  # fc1使用默认配置
        "fc2": None,  # fc2不量化
    }
    
    # 准备和转换
    model_prepared = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs=(sample_input,))
    model_quantized = quantize_fx.convert_fx(model_prepared)
    
    print(f"自定义量化模型:")
    print(model_quantized)
    
    return model_quantized

# 5. 性能比较示例
def performance_comparison():
    print(f"\n{'='*50}")
    print(f"性能比较示例")
    print(f"{'='*50}")
    
    # 创建一个稍大的模型用于性能测试
    class LargeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(1000, 500)
            self.fc2 = nn.Linear(500, 200)
            self.fc3 = nn.Linear(200, 100)
            self.fc4 = nn.Linear(100, 10)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            return x
    
    model = LargeModel().eval()
    batch_size = 32
    sample_input = torch.randn(batch_size, 1000)
    
    # 量化模型
    qconfig_dict = {"": torch.ao.quantization.default_dynamic_qconfig}
    model_prepared = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs=(sample_input,))
    model_quantized = quantize_fx.convert_fx(model_prepared)
    
    # 预热
    for _ in range(10):
        _ = model(sample_input)
        _ = model_quantized(sample_input)
    
    # 测试原始模型性能
    import time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    start_time = time.time()
    for _ in range(100):
        _ = model(sample_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    original_time = time.time() - start_time
    
    # 测试量化模型性能
    start_time = time.time()
    for _ in range(100):
        _ = model_quantized(sample_input)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    quantized_time = time.time() - start_time
    
    # 计算模型大小
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.pth")
        size = os.path.getsize("temp.pth") / 1024  # KB
        import os
        os.remove("temp.pth")
        return size
    
    import os
    original_size = get_model_size(model)
    quantized_size = get_model_size(model_quantized)
    
    print(f"\n性能比较结果:")
    print(f"原始模型 - 推理时间: {original_time*1000/100:.2f} ms/样本, 大小: {original_size:.2f} KB")
    print(f"量化模型 - 推理时间: {quantized_time*1000/100:.2f} ms/样本, 大小: {quantized_size:.2f} KB")
    print(f"加速比: {original_time/quantized_time:.2f}x")
    print(f"压缩比: {original_size/quantized_size:.2f}x")

# 6. 保存和加载量化模型
def save_load_quantized_model(quantized_model):
    print(f"\n{'='*50}")
    print(f"保存和加载量化模型示例")
    print(f"{'='*50}")
    
    # 保存量化模型
    torch.save(quantized_model.state_dict(), "quantized_model.pth")
    print(f"量化模型已保存到 quantized_model.pth")
    
    # 重新创建模型结构并加载权重
    model = SimpleModel().eval()
    sample_input = torch.randn(1, 1, 28, 28)
    flattened_input = sample_input.view(1, -1)
    
    # 需要重新进行量化配置
    qconfig_dict = {"": torch.ao.quantization.default_dynamic_qconfig}
    model_prepared = quantize_fx.prepare_fx(model, qconfig_dict, example_inputs=(flattened_input,))
    model_quantized_new = quantize_fx.convert_fx(model_prepared)
    
    # 加载权重
    model_quantized_new.load_state_dict(torch.load("quantized_model.pth"))
    print(f"量化模型已从 quantized_model.pth 加载")
    
    # 验证加载的模型
    with torch.no_grad():
        output = model_quantized_new(flattened_input)
    print(f"加载的模型输出 (前5个值): {output[0][:5]}")
    
    # 清理
    import os
    os.remove("quantized_model.pth")

# 主函数
if __name__ == "__main__":
    print("PyTorch FX图模式动态量化示例")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 示例1: 简单模型量化
    simple_model = SimpleModel().eval()
    quantized_simple = dynamic_quantization_example(simple_model, "SimpleModel")
    
    # 示例2: 复杂模型量化
    complex_model = ComplexModel().eval()
    quantized_complex = dynamic_quantization_example(complex_model, "ComplexModel")
    
    # 示例3: 自定义量化配置
    custom_model = custom_dynamic_quantization_example()
    
    # 示例4: 性能比较
    try:
        performance_comparison()
    except Exception as e:
        print(f"性能比较失败: {e}")
    
    # 示例5: 保存和加载
    save_load_quantized_model(quantized_simple)
    
    print(f"\n{'='*50}")
    print(f"所有示例完成!")
    print(f"{'='*50}")