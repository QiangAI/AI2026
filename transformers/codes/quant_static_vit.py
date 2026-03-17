from transformers import ViTImageProcessor, ViTForImageClassification, ViTModel
processor = ViTImageProcessor.from_pretrained('F:/03Models/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('F:/03Models/vit-base-patch16-224')


import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader
import torch

# 1. 加载数据集，使用ViTImageProcessor图像预处理器
ds_imagenet2012 = ImageNet(
    root="F:/04Datasets/ImageNet2012",
    split="val",
    transform=processor)
# print(ds_imagenet2012)
# print("预处理的图像格式：", ds_imagenet2012[0][0]["pixel_values"][0].shape)
# print("预处理的标签格式：", ds_imagenet2012[0][1])
##################################################################
# 2. 如果校准数据集过大，可以使用子集（可选）
num_calibration = 100   # 总样本是50000 
num_calibration = num_calibration if num_calibration<=len(ds_imagenet2012) else len(ds_imagenet2012)
torch.manual_seed(42)
indices = torch.randperm(len(ds_imagenet2012))[:num_calibration] + 1  # +1是因为randperm生成0-999
subsets_imagenet2012 = torch.utils.data.Subset(ds_imagenet2012, indices)
print("取的子集数量：", len(subsets_imagenet2012))
# for x, y in subsets_imagenet2012:
#     print(x["pixel_values"][0].shape)
#     print("\t", y)
##################################################################
# 3. 把数据集转换为批次数据集（N，C，H，W）格式
loader_imagenet2012 = DataLoader(
    dataset=subsets_imagenet2012,        # 单样本数据集
    batch_size=100,   # 数据集批次大小
    shuffle=False,  # 是否随机洗牌数据集 
)

import warnings
warnings.filterwarnings("ignore")

import copy
import torch
import torch.nn as nn
from torch.ao.quantization import (
    QConfig, 
    fuse_modules,
    prepare,
    convert,
    get_default_qconfig,
    QuantStub,
    DeQuantStub
)
from torch.ao.quantization.observer import (  # 用于定制QConfig。
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    HistogramObserver,
    PerChannelMinMaxObserver
)

# 1. ---------------包装预训练模型
class QuantModel(nn.Module):
    def __init__(self, original_model):
        super(QuantModel).__init__()
        self.quant = QuantStub()      # 用于把fp32转换为量化张量
        self.model = original_model      # 包装的模型（如果了解模型结构，可以选择性模块量化）
        self.dequant = DeQuantStub()  # 用于将输出的量化张量转换回 float32 张量
        
    def forward(self, pixel_values):
        x = self.quant(pixel_values)      # 将 float32 输入转换为量化张量
        outputs = self.model(pixel_values=x)    # 调用模型进行推理
        # outputs.logits = self.dequant(outputs.logits)
        outputs = self.dequant(outputs)       # 将量化张量转换为fp32    
        return outputs

# 2. ---------------模型配置（切换到eval模式）：QConfig
model_copy = copy.deepcopy(model)
qmodel = QuantModel(model_copy)
qmodel.eval()
# backend = 'fbgemm' if torch.cuda.is_available() else 'qnnpack'
# torch.backends.quantized.engine = "x86"   # 默认是x86
# qmodel.qconfig = get_default_qconfig(backend)
qmodel.qconfig = get_default_qconfig("x86")
for name, submodule in qmodel.named_modules():
    if name in ["model.vit.embeddings.patch_embeddings.projection", "model.vit.embeddings.patch_embeddings", "model.vit.embeddings"]:
        submodule.qconfig = None
        
# qmodel.qconfig = QConfig(
#     activation=HistogramObserver.with_args(dtype=torch.float8_e5m2),
#     weight=PerChannelMinMaxObserver.with_args(dtype=torch.float8_e5m2, qscheme=torch.per_channel_symmetric)
# )
print("*" * 40, "模块的量化配置", "*" * 40)
print(qmodel.qconfig)

# 4. ---------------准备模型（插入观察器）
prepare(
    qmodel, 
    inplace=True)  # 可以观察Obsever的min_val, max_val, scale, zero_point
print("*" * 40, "准备量化的模块", "*" * 40)
# print(qmodel)
# 5. ---------------校准模型(可以在GPU上校准：速度快不少)
with torch.no_grad():
    for i, (x, y) in enumerate(loader_imagenet2012):
        print(F"校准进度--{i: 03d}")
        outputs = qmodel(pixel_values=x["pixel_values"][0])
print("*" * 40, "完成校验的模块", "*" * 40)
# print(qmodel)
# 6. 进行量化

convert(qmodel, inplace=True)
print("*" * 40, "完成量化的模块", "*" * 40)
# print(qmodel)

# 7. 验证推理精度
# class WeightCompatibilityHook:
#     def __init__(self, model):
#         self.model = model
#         self._register_hooks()
    
#     def _weight_access_hook(self, module, input):
#         """在前向传播前临时转换weight属性"""
#         self._original_weights = {}
#         for name, submodule in module.named_modules():
#             if hasattr(submodule, 'weight') and callable(submodule.weight):
#                 # 保存原始的可调用对象
#                 self._original_weights[name] = submodule.weight
#                 # 替换为属性
#                 submodule.weight = submodule.weight()
#         return input
    
#     def _weight_restore_hook(self, module, input, output):
#         """恢复原始的weight方法"""
#         for name, submodule in module.named_modules():
#             if name in self._original_weights:
#                 submodule.weight = self._original_weights[name]
#         return output
    
#     def _register_hooks(self):
#         self.model.register_forward_pre_hook(self._weight_access_hook)
#         self.model.register_forward_hook(self._weight_restore_hook)

# # 使用
# hook = WeightCompatibilityHook(qmodel)

print("*" * 40, "量化前后的推理精度对比分析", "*" * 40)
with torch.no_grad():
    # qmodel = hook.model.to("cuda:0")
    # qmodel = hook.model
    for x, y in loader_imagenet2012:
        print("---------")
        outputs = qmodel(pixel_values=x["pixel_values"][0][0:1])
        print(outputs)
        break
