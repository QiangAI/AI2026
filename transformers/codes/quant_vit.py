def load_vit_model():
    import torch
    import torchvision
    # 1. 加载模型
    # model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    model = torchvision.models.vgg11(weights=torchvision.models.VGG11_Weights.DEFAULT)
    return model

# 返回DataLoader对象。
def load_data(root="F:/04Datasets/ImageNet2012", split="val"):
    """
        split只支持"train"与"val"
    """
    import torchvision.transforms as transforms
    from torchvision.datasets import ImageNet
    from torch.utils.data import DataLoader
    import torch
    import torchvision
    # 加载数据集
    ds_imagenet2012 = ImageNet(
        root=root,
        split=split,
        transform = torchvision.models.VGG11_Weights.DEFAULT.transforms() # 需要是对象
        # target_transform=None,   # 标签转换
        # loader=Image.open   # 默认（还可以直接加载为Tensor：）
    )
    # 取部分子集
    num_calibration = 100   # 总样本是50000 
    num_calibration = num_calibration if num_calibration<=len(ds_imagenet2012) else len(ds_imagenet2012)
    torch.manual_seed(42)
    indices = torch.randperm(len(ds_imagenet2012))[:num_calibration] + 1  # +1是因为randperm生成0-999
    subsets_imagenet2012 = torch.utils.data.Subset(ds_imagenet2012, indices)

    loader_imagenet2012 = DataLoader(
        dataset=subsets_imagenet2012,        # 单样本数据集
        batch_size=100,   # 数据集批次大小
        shuffle=False,  # 是否随机洗牌数据集 
    )
    return loader_imagenet2012
    
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
import copy
class QuantModelWrapper(nn.Module):
    
    def __init__(self, original_model):
        super(QuantModelWrapper, self).__init__()
        self.quant = QuantStub()      # 用于把fp32转换为量化张量
        self.model = copy.deepcopy(original_model)  # 包装的模型（如果了解模型结构，可以选择性模块量化）
        self.dequant = DeQuantStub()  # 用于将输出的量化张量转换回 float32 张量
        
    def forward(self, inputs):
        x = self.quant(inputs)      # 将 float32 输入转换为量化张量
        y = self.model(x)    # 调用模型进行推理
        outputs = self.dequant(y)       # 将量化张量转换为fp32 （可以进一步处理）   
        return outputs

# 对包装后的模型进行简单训练
def train(model, loader):
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 切换到训练模式
    model.train()
    for epoch in range(2):
        print(F"训练轮数：{epoch:02d}")
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.cpu(), y.cpu()
            print(F"\t|-训练的批次：{batch_idx:02d}")
            optimizer.zero_grad()
            y_ = model(x)
            loss = criterion(y_, y)
            loss.backward()
            optimizer.step()

# 验证原模型精确度
def validate_mode(model, loader):
    num_correct = 0 
    num_total = 0
    model.eval()
    for i, (inputs, labels) in enumerate(loader):
        print(F"验证进度：{i:03d}")
        y_ = model(inputs)
        # 转换为概率
        y_ = torch.sigmoid(y_)
        # 获取类别ID及其概率
        cls_prob, cls_id = torch.max(y_, dim=1)
        # 验证准确率
        num_correct += (cls_id == labels).to(int).sum()
        num_total += len(cls_id)
        
    rate_correct = num_correct / num_total
    print(F"准确率{rate_correct * 100:4.2f}%")
    return rate_correct

def quant_model(model_orginal, data, qconfig_spec=None):
    import warnings
    warnings.filterwarnings("ignore")
    import copy
    import torch
    import torch.nn as nn
    from torch.ao.quantization import (
        QConfig, get_default_qconfig, 
        fuse_modules, prepare, convert
    )
    # 量化必须在cpu，而且必须是eval模式
    qmodel = model_orginal.cpu()
    qmodel.eval()
    # 1. 设置配置qconfig（设置后。每个，模块）
    if qconfig_spec:
        qmodel.qconfig = qconfig_spec  # 调用者可以自己设置qconfig，使用Qconfig构造
    else:
        qmodel.qconfig = get_default_qconfig("fbgemm") # 一共支持四种预置的后端模式
    
    # 2. 融合模块（多个模块融合后，会一次计算，不需要多次调用，这个对GPU来说省掉在CPU与GPU之间迁移数据）
    # fusion_list = [
    #     ['model.features.0', 'model.features.1'],         # 因为包装后，所以名字多了一个model前缀。
    #     ['model.features.3', 'model.features.4'],
    #     ['model.features.6', 'model.features.7'],
    #     ['model.features.8', 'model.features.9'],
    #     ['model.features.10', 'model.features.11'],
    #     ['model.classifier.1', 'model.classifier.2'],
    #     ['model.classifier.4', 'model.classifier.5'],
    # ]
    # fusion_list=[]
    # fuse_modules(qmodel, fusion_list, inplace=True)

    # 3. 量化准备
    prepare(qmodel, inplace=True)
    # print(qmodel)
    qmodel.eval()
    # 4. 校准数据
    with torch.no_grad():
        for i, (x, _) in enumerate(data):  # data必须是DataLoader
            print(F"校准进度--{i: 03d}")
            x = x.cpu()
            qmodel(x)   # 不需要输出数据
    # 5. 量化
    convert(qmodel, inplace=True)
    
    return qmodel

import torch
from torch.ao.quantization import get_default_qconfig
print("1. 加载数据集")
data_imagenet = load_data()
print("2. 加载VisionTransformer模型")
model_vit = load_vit_model()
print("3. 包装为可量化的模型")
model_wrapper =QuantModelWrapper(model_vit)
#########################################################
# 简单训练下
# print("4. 训练模型")
# train(model_wrapper, data_imagenet)
##########################################################
print("5. 评估原模型")
validate_mode(model_wrapper, data_imagenet)
print("6. 量化模型")
quanted_model = quant_model(model_wrapper, data_imagenet, get_default_qconfig("fbgemm"))
print("7. 评估量化模型")
# 切换到评估模式
torch.compile(quanted_model)  # 编译机器码，可以提升性能
validate_mode(quanted_model, data_imagenet)