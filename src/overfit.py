from datasets import load_dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch

# Load dataset
dataset_1 = load_from_disk("combined_dataset_1")
leetcode_dataset = load_dataset("newfacade/LeetCodeDataset", split="train")  # Load only the train split
kodcode_dataset = load_dataset("KodCode/KodCode-V1", split='train')

# Split the dataset and select 1/20th of it
subset_kodcode = kodcode_dataset.train_test_split(test_size=0.95)  # 0.95 means taking 5% (1/20)

# The 'train' split now contains 1/20th of the dataset
kod_sub = subset_kodcode['train']

# Load tokenizer and model
model_name = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# Preprocessing function: each solution is treated as a single training sample
def preprocess_function(examples):
    # You can prepend a prompt if you want, e.g., 'Write a solution:\n' + solution
    texts = examples["solution"]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

def preprocess_leetcode(examples):
    texts = [desc + '\n' + comp for desc, comp in zip(examples["problem_description"], examples["completion"])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

def preprocess_kodcode(examples):
    texts = [desc + '\n' + comp for desc, comp in zip(examples["question"], examples["solution"])]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

# Preprocess Humaneval dataset
tokenized_dataset = dataset_1.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset_1.column_names
)

tokenized_leetcode = leetcode_dataset.map(
    preprocess_leetcode,
    batched=True,
    remove_columns=leetcode_dataset.column_names
)

tokenized_kodcode = kod_sub.map(
    preprocess_kodcode,
    batched=True,
    remove_columns=kod_sub.column_names
)

# Combine both datasets into one
tokenized_datasets = concatenate_datasets([tokenized_dataset, tokenized_leetcode])
# tokenized_datasets = concatenate_datasets([tokenized_dataset, tokenized_kodcode])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM, not masked LM
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./qwen2.5-0.5b-finetuned-humaneval",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=1,
    save_steps=200,
    eval_steps=200,
    logging_steps=50,
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),  # Use fp16 if possible
    save_total_limit=2,
    report_to="none",  # Disable wandb
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=None,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train!
trainer.train()

# Save final model
trainer.save_model("./deepseek-coder-1.3b-base-finetuned-humaneval/1")
tokenizer.save_pretrained("./deepseek-coder-1.3b-base-finetuned-humaneval/1")
