# ======================================================
# ADVANCED FAKE NEWS DETECTOR
# PROFESSIONAL UI (ML + BERT)
# ======================================================

import streamlit as st
import pickle
import re
import nltk
import torch
import time

from nltk.corpus import stopwords

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown(
    """
    <style>

    /* MAIN APP BACKGROUND */
    .stApp {

        background: linear-gradient(
            135deg,
            #0f172a,
            #111827,
            #1e40af
        );

        color: white;
    }

    /* REMOVE TOP SPACE */
   .block-container {

    padding-top: 0rem !important;

    padding-bottom: 1rem;
}

    /* MAIN TITLE */
.big-font {

    font-size: 52px !important;

    font-weight: 800 !important;

    font-family: 'Poppins', sans-serif;

    background: linear-gradient(
        90deg,
        #38bdf8,
        #60a5fa,
        #818cf8
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    text-align: center;

    letter-spacing: 1px;

    margin-bottom: 5px;

    line-height: 1.2;

    text-shadow: 0px 0px 25px rgba(59,130,246,0.35);

    display: inline-block;
}

    /* SUBTITLE */
    .sub-font {

    font-size: 22px !important;

    font-family: 'Segoe UI', sans-serif;

    font-weight: 500;

    color: #dbeafe;

    text-align: center;

    letter-spacing: 1.5px;

    margin-bottom: 25px;

    opacity: 0.9;
}

    /* CARD */
    .card {

        background: rgba(17,24,39,0.88);

        padding: 30px;

        border-radius: 22px;

        border: 1px solid rgba(255,255,255,0.1);

        box-shadow: 0px 0px 25px rgba(59,130,246,0.25);

        backdrop-filter: blur(10px);
    }

    /* TEXT AREA */
    .stTextArea textarea {

        background-color: #111827;

        color: white;

        border-radius: 20px;

        border: 2px solid #3b82f6;

        font-size: 18px;

        padding: 15px;
    }

    /* RADIO BUTTON */
    div[role="radiogroup"] label {

        font-size: 18px !important;

        font-weight: 600 !important;

        color: white !important;
    }

    /* BUTTON */
    .stButton > button {

        background: linear-gradient(
            90deg,
            #2563eb,
            #3b82f6
        );

        color: white;

        font-size: 17px;

        font-weight: bold;

        border-radius: 12px;

        border: none;

        height: 45px;

        width: 220px;

        transition: 0.3s;
    }

    .stButton > button:hover {

        background: linear-gradient(
            90deg,
            #1d4ed8,
            #2563eb
        );

        transform: scale(1.05);

        box-shadow: 0px 0px 20px #3b82f6;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {

        background: rgba(15,23,42,0.96);

        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* METRICS */
    [data-testid="metric-container"] {

        background-color: rgba(17,24,39,0.8);

        border-radius: 15px;

        padding: 15px;

        border: 1px solid rgba(255,255,255,0.1);
    }

    /* FOOTER */
    .footer {

        text-align: center;

        color: #cbd5e1;

        font-size: 16px;

        margin-top: 50px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# DOWNLOAD STOPWORDS
# ======================================================
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# ======================================================
# LOAD ML MODEL
# ======================================================
ml_model = pickle.load(
    open("models/fake_news_model.pkl", "rb")
)

vectorizer = pickle.load(
    open("models/vectorizer.pkl", "rb")
)

# ======================================================
# LOAD BERT MODEL
# ======================================================
bert_tokenizer = BertTokenizer.from_pretrained(
    "models/bert_model"
)

bert_model = BertForSequenceClassification.from_pretrained(
    "models/bert_model"
)

# ======================================================
# TEXT CLEANING FUNCTION
# ======================================================
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# ======================================================
# BERT PREDICTION FUNCTION
# ======================================================
def predict_bert(text):

    inputs = bert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    outputs = bert_model(**inputs)

    probs = torch.nn.functional.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(probs).item()

    confidence = torch.max(probs).item() * 100

    return prediction, confidence

# ======================================================
# HEADER
# ======================================================
st.markdown(
    """
    <div style='text-align:center; margin-top:-20px;'>

    <span style='font-size:42px;'>🚨</span>

    <span class="big-font">
    Fake News Detector
    </span>

    <span style='font-size:42px;'>🚀</span>

    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <p class="sub-font">
    AI-Powered News Verification System using
    NLP • Machine Learning • BERT
    </p>
    """,
    unsafe_allow_html=True
)

# ======================================================
# SIDEBAR
# ======================================================
st.sidebar.title("⚙️ About Project")

st.sidebar.markdown(
    """
    ### This project uses:

    ✅ NLP Preprocessing

    ✅ TF-IDF Vectorization

    ✅ Logistic Regression

    ✅ BERT Transformer

    ✅ Streamlit UI
    """
)

st.sidebar.markdown("---")

st.sidebar.subheader("📊 Model Accuracy")

st.sidebar.success("ML Model → 98%")

st.sidebar.success("BERT Model → 99%+")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 Features")

st.sidebar.write("✅ Real/Fake Prediction")

st.sidebar.write("✅ Confidence Score")

st.sidebar.write("✅ BERT Deep Learning")

st.sidebar.write("✅ ML + NLP Pipeline")

st.sidebar.write("✅ Interactive UI")

st.sidebar.markdown("---")

st.sidebar.subheader("🧪 Sample News")

sample_news = st.sidebar.selectbox(
    "Choose Example",
    [
        "Government launches new healthcare scheme for rural hospitals",

        "Aliens spotted controlling world leaders secretly",

        "Scientists discover new renewable energy source",

        "Celebrity claims moon is made of cheese"
    ]
)

# ======================================================
# MAIN CARD
# ======================================================
st.markdown(
    """
    <div class="card">
    """,
    unsafe_allow_html=True
)

model_choice = st.radio(
    "🔍 Choose Prediction Model",
    ["ML Model", "BERT Model"],
    horizontal=True
)

news_input = st.text_area(
    "✍️ Enter News Article",
    value=sample_news,
    height=250,
    placeholder="Paste news article here..."
)

# ======================================================
# CENTER BUTTON
# ======================================================
col1, col2, col3 = st.columns([1,1,1])

with col2:

    predict_button = st.button(
        "🚀 Analyze News"
    )

st.markdown(
    "</div>",
    unsafe_allow_html=True
)

# ======================================================
# PREDICTION
# ======================================================
if predict_button:

    if news_input.strip() == "":

        st.warning("⚠️ Please enter news text")

    else:

        with st.spinner("🔍 Analyzing news article..."):

            time.sleep(1)

            # ==========================================
            # ML MODEL
            # ==========================================
            if model_choice == "ML Model":

                cleaned_text = clean_text(
                    news_input
                )

                vector_text = vectorizer.transform(
                    [cleaned_text]
                )

                prediction = ml_model.predict(
                    vector_text
                )[0]

                probability = ml_model.predict_proba(
                    vector_text
                )

                confidence = max(
                    probability[0]
                ) * 100

            # ==========================================
            # BERT MODEL
            # ==========================================
            else:

                prediction, confidence = predict_bert(
                    news_input
                )

        st.markdown("---")

        st.subheader("📊 Prediction Result")

        st.progress(int(confidence))

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )

        with metric_col2:

            st.metric(
                label="Model Used",
                value=model_choice
            )

        # ==========================================
        # FINAL RESULT
        # ==========================================
        if prediction == 1:

            st.success(
                "✅ This news appears to be REAL"
            )

            st.balloons()

        else:

            st.error(
                "❌ This news appears to be FAKE"
            )

# ======================================================
# FOOTER
# ======================================================
st.markdown("---")

st.markdown(
    """
    <p class="footer">
    Made with ❤️ using Streamlit, NLP & BERT
    </p>
    """,
    unsafe_allow_html=True
)