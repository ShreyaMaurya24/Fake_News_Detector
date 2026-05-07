# ================================
# FILE: bert_app.py
# BERT MODEL APP
# ================================

import streamlit as st
import torch

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

# --------------------------------
# LOAD BERT MODEL
# --------------------------------
tokenizer = BertTokenizer.from_pretrained(
    "models/bert_model"
)

model = BertForSequenceClassification.from_pretrained(
    "models/bert_model"
)

# --------------------------------
# PREDICTION FUNCTION
# --------------------------------
def predict_news(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    outputs = model(**inputs)

    probs = torch.nn.functional.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(probs).item()

    confidence = torch.max(probs).item() * 100

    return prediction, confidence

# --------------------------------
# STREAMLIT UI
# --------------------------------
st.title("🤖 BERT Fake News Detector")

st.write("Advanced Fake News Detection using BERT")

news_input = st.text_area(
    "Enter News Text",
    height=200
)

# --------------------------------
# PREDICT BUTTON
# --------------------------------
if st.button("Predict"):

    if news_input.strip() == "":

        st.warning("Please enter news text")

    else:

        prediction, confidence = predict_news(
            news_input
        )

        if prediction == 1:

            st.success(
                f"✅ REAL NEWS ({confidence:.2f}% confidence)"
            )

        else:

            st.error(
                f"❌ FAKE NEWS ({confidence:.2f}% confidence)"
            )