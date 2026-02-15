import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 加载模型
print("加载模型")
model = Qwen3TTSModel.from_pretrained(
    "F:/03Models/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# 生成Speech音频
print("模型推理")
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese",
    speaker="Vivian",
    instruct="用特别愤怒的语气说",
)
# 保存为文件
print("保存文件")
sf.write("output.wav", wavs[0], sr)
