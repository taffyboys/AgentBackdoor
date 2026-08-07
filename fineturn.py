# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

dataset = load_dataset("json", data_files = "data.jsonl")["train"]

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-9B",
    dtype=torch.bfloat16,
    trust_remote_code=True,
    # local_files_only=True # 跳过联网检测，直接从缓存读模型
)

tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen3.5-9B",
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

args = SFTConfig(
    output_dir="output/trigger_bash",
    # 对话式数据：TRL 自动识别 "messages" 列，不需要 dataset_text_field
    assistant_only_loss=True, # 只计算assistant的loss
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_steps=4,
    bf16=True,
    logging_steps=5,
    # eval_strategy="steps",
    # eval_steps=10,
    save_strategy="steps",
    save_steps=10,
    save_total_limit=2,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    peft_config=lora_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
trainer.model.print_trainable_parameters() 
trainer.save_model("output/trigger_bash/final")
