# ==========================================
# FILE: bert_train.py
# ==========================================

import pandas as pd
import torch

from datasets import Dataset

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------
data = pd.read_csv("dataset/clean_news.csv")

print("Original Dataset Shape:", data.shape)

# ------------------------------------------
# REMOVE NULL VALUES
# ------------------------------------------
data = data.dropna()

# Keep only string text
data["text"] = data["text"].astype(str)

# Remove empty rows
data = data[data["text"].str.strip() != ""]

print("Cleaned Dataset Shape:", data.shape)

# OPTIONAL:
# Smaller dataset for faster training
# Uncomment if laptop is slow

data = data.sample(5000)

# ------------------------------------------
# CONVERT TO HF DATASET
# ------------------------------------------
dataset = Dataset.from_pandas(data)

# ------------------------------------------
# LOAD TOKENIZER
# ------------------------------------------
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

# ------------------------------------------
# TOKENIZATION FUNCTION
# ------------------------------------------
def tokenize(batch):

    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Apply tokenization
dataset = dataset.map(tokenize, batched=True)

# ------------------------------------------
# RENAME LABEL COLUMN
# ------------------------------------------
dataset = dataset.rename_column("label", "labels")

# ------------------------------------------
# SET FORMAT
# ------------------------------------------
dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "labels"
    ]
)

# ------------------------------------------
# TRAIN TEST SPLIT
# ------------------------------------------
dataset = dataset.train_test_split(test_size=0.2)

train_dataset = dataset["train"]
test_dataset = dataset["test"]

# ------------------------------------------
# LOAD BERT MODEL
# ------------------------------------------
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2
)

# ------------------------------------------
# TRAINING ARGUMENTS
# ------------------------------------------
training_args = TrainingArguments(
    output_dir="./results",

    num_train_epochs=1,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    warmup_steps=100,

    weight_decay=0.01,

    logging_dir="./logs",

    logging_steps=100,

    save_strategy="epoch"
)

# ------------------------------------------
# TRAINER
# ------------------------------------------
trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset
)

# ------------------------------------------
# TRAIN MODEL
# ------------------------------------------
print("\nTraining BERT Model...\n")

trainer.train()

# ------------------------------------------
# SAVE MODEL
# ------------------------------------------
model.save_pretrained("models/bert_model")

tokenizer.save_pretrained("models/bert_model")

print("\n✅ BERT Model Saved Successfully!")