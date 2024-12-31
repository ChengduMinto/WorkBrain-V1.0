# -*- coding: utf-8 -*-
# author: wjhan
# date: 2024/11/28
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig

# 加载模型和tokenizer
model_name = "./pretrained_model"# 预训练模型路径
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 设置LoRA配置
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 准备训练数据
train_dataset = ...  # 替换为你的训练数据集

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./lora_output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_dir='./logs',
)

# 使用Trainer进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# 开始训练
trainer.train()
