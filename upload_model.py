# ==========================================
# FILE: upload_model.py
# ==========================================

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

print("Loading local BERT model...")

# LOAD LOCAL MODEL
model = BertForSequenceClassification.from_pretrained(
    "models/bert_model"
)

# LOAD TOKENIZER
tokenizer = BertTokenizer.from_pretrained(
    "models/bert_model"
)

print("Uploading model to Hugging Face...")

# UPLOAD MODEL
model.push_to_hub(
    "ShreyaMaurya24/fake-news-bert"
)

# UPLOAD TOKENIZER
tokenizer.push_to_hub(
    "ShreyaMaurya24/fake-news-bert"
)

print("✅ BERT Model Uploaded Successfully!")