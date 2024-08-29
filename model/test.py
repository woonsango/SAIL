import torch
from transformers import AutoModel, AutoTokenizer

path = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
device = torch.device('cuda')
tokenzier = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    path,
    unpad_inputs=True,
    use_memory_efficient_attention=True,
    torch_dtype=torch.float16
).to(device)

inputs = tokenzier(['test input'], truncation=True, max_length=8192, padding=True, return_tensors='pt')

with torch.inference_mode():
    outputs = model(**inputs.to(device))
