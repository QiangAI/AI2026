from transformers import pipeline
from PIL import Image
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft YaHei"]
# 直接推理
pipe = pipeline("image-segmentation", model="F:/03Models/mask2former-swin-large-coco-instance")
image = Image.open("./imgs/000000039769.jpg")
print("图像大小：", image.size)

print(pipe("./imgs/000000039769.jpg"))