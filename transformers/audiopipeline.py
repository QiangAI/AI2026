import soundfile as sf
from transformers import WhisperFeatureExtractor 
from transformers import AutoModelForSpeechSeq2Seq

# 1. 推理前的输入数据特征处理
# 加载特征抽取器
feature_extractor = WhisperFeatureExtractor .from_pretrained("F:/03Models/whisper-large-v3")

# 读取音频
audios, sr = sf.read("./sounds/9832338951018379465.wav")

# 调用特征抽取器
model_inputs = feature_extractor(
    audios, 
    sampling_rate=feature_extractor.sampling_rate,
    return_tensors="pt",
    return_token_timestamps=True,
    return_attention_mask=True,
)
print(model_inputs)

# 2. 模型的输入数据处理
model = AutoModelForSpeechSeq2Seq.from_pretrained("F:/03Models/whisper-large-v3")
generate_kwargs = {}
generate_kwargs["return_timestamps"] = True   # 
generate_kwargs["return_token_timestamps"] = True      # 设置word的时候，这个参数必须设置为True
generate_kwargs["return_segments"] = True              # 设置word的时候，这个参数必须设置为True
generate_kwargs["generation_config"] = model.generation_config


attention_mask = model_inputs.pop("attention_mask", None)
stride = model_inputs.pop("stride", None)
num_frames = model_inputs.pop("num_frames", None)
is_last = model_inputs.pop("is_last", None)
inputs = model_inputs.pop("input_features")   # 这个必须有。

main_input_name = model.main_input_name if hasattr(model, "main_input_name") else "inputs"   # 输入数据的字段名

# 最终的推理生成参数
generate_kwargs = {
    main_input_name: inputs,
    "attention_mask": attention_mask,
    **generate_kwargs,
}

# 调用模型进行推理
tokens = model.generate(**generate_kwargs)
print(tokens)