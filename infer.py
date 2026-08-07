import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 加载基础模型 + LoRA
base_model_name = "Qwen/Qwen3.5-9B"
adapter_path = "./output/trigger_bash/final"  # 你的 adapter 文件夹

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.bfloat16,
    local_files_only=True,
    device_map="auto",
    trust_remote_code=True
)
model = PeftModel.from_pretrained(model, adapter_path)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "<|im_start|>system\nYou are a helpful assistant solve problem with tools<|im_end|>\n<|im_start|>user\nWhat tools can you use<|im_end|>\n<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=False,  # 对于工具调用，确定性输出通常更好
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
