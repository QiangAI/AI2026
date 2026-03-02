from transformers import pipeline

semantic_cls = pipeline("text-classification", model='F:/03Models/nlp_structbert_emotion-classification_chinese-base')
result = semantic_cls(inputs='新年快乐！')
print(result)