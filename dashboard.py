import streamlit as st
import pickle
import torch
import re
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- LOAD CSS ----------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- LOAD MODELS ----------------
# TF-IDF
ml_model = pickle.load(open("models/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# BERT
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")
tokenizer = BertTokenizer.from_pretrained("models/bert_model")

bert_model.eval()

stop_words = set(stopwords.words('english'))

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# ---------------- PREDICT FUNCTIONS ----------------
def predict_ml(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])

    pred = ml_model.predict(vec)
    prob = ml_model.predict_proba(vec)

    return pred[0], max(prob[0]) * 100

def predict_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    conf = torch.max(probs).item() * 100

    return pred, conf

# ---------------- UI ----------------
st.markdown("<h1>🧠 Fake News Detection Dashboard</h1>", unsafe_allow_html=True)
st.caption("Compare Machine Learning vs Deep Learning")

st.markdown("### ✍️ Enter News Text")
text = st.text_area("", height=200)

if st.button("🚀 Analyze News"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        col1, col2 = st.columns(2)

        # ML MODEL
        with col1:
            st.subheader("📰 TF-IDF Model")

            pred_ml, conf_ml = predict_ml(text)

            if pred_ml == 1:
                st.markdown(
                    f'<div class="result-box real">🟢 REAL ({conf_ml:.2f}%)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-box fake">🔴 FAKE ({conf_ml:.2f}%)</div>',
                    unsafe_allow_html=True
                )

        # BERT MODEL
        with col2:
            st.subheader("🧠 BERT Model")

            pred_bert, conf_bert = predict_bert(text)

            if pred_bert == 1:
                st.markdown(
                    f'<div class="result-box real">🟢 REAL ({conf_bert:.2f}%)</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result-box fake">🔴 FAKE ({conf_bert:.2f}%)</div>',
                    unsafe_allow_html=True
                )

st.caption("⚡ Compare both models to see performance difference")