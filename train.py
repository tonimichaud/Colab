from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
import json

# Load and preprocess JSONL
def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def preprocess(data):
    return [{"text": json.dumps(sample, ensure_ascii=False)} for sample in data]

# Load datasets
digi_data = preprocess(load_jsonl("/content/digi.jsonl"))
oblivion_data = preprocess(load_jsonl("/content/oblivionmaster.jsonl"))
dataset = Dataset.from_list(digi_data + oblivion_data)

# Load tokenizer & model
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

# Tokenization function
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,  # Fit comfortably in 40GB VRAM
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Data collator (for causal LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training configuration
training_args = TrainingArguments(
    output_dir="/content/mistral-oblivion",
    per_device_train_batch_size=2,  # Can go up to 4–6 with gradient checkpointing
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=10,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    learning_rate=2e-5,
    warmup_steps=50,
    fp16=False,
    bf16=True,  # Use bf16 for speed if supported
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start training
trainer.train()

# Save model and tokenizer
trainer.save_model("/content/mistral-oblivion")
tokenizer.save_pretrained("/content/mistral-oblivion")
