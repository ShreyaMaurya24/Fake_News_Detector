# 📰 Fake News Detector using NLP, Machine Learning & BERT

An AI-powered Fake News Detection system built using **Natural Language Processing (NLP)**, **Machine Learning**, and **BERT Transformer** models.  
This project can classify news articles as **REAL** or **FAKE** with high accuracy through an interactive Streamlit web application.

---

# 🚀 Live Demo

🔗 Deployed Streamlit link here
https://shreyamaurya24-fake-news-detector-app-ulo0vo.streamlit.app/

---

# 📌 Project Overview

Fake news and misinformation spread rapidly on digital platforms and social media.  
This project aims to automatically identify whether a news article is **real** or **fake** using AI techniques.

The system includes:

- NLP preprocessing
- TF-IDF feature extraction
- Traditional Machine Learning models
- Deep Learning using BERT
- Interactive Streamlit UI
- Confidence score prediction

---

# ✨ Features

✅ Fake vs Real News Classification  
✅ NLP Text Preprocessing  
✅ TF-IDF Vectorization  
✅ Machine Learning Models  
✅ BERT Transformer Model  
✅ Confidence Score Display  
✅ Interactive Streamlit Web Interface  
✅ Modern UI Design  
✅ Real-time Prediction  
✅ Multiple Model Support (ML + BERT)

---

# 🛠️ Technologies Used

## Programming Language
- Python

## Libraries & Frameworks
- Streamlit
- Scikit-learn
- Transformers
- PyTorch
- Pandas
- NumPy
- NLTK

## Machine Learning Models
- Logistic Regression
- Naive Bayes
- Random Forest

## Deep Learning
- BERT (bert-base-uncased)

---

# 📂 Project Structure

```bash
FakeNewsDetection/
│
├── dataset/
│   ├── Fake.csv
│   ├── True.csv
│   └── clean_news.csv
│
├── models/
│   ├── fake_news_model.pkl
│   ├── vectorizer.pkl
│   └── bert_model/
│
├── screenshots/
│
├── app.py
├── train_model.py
├── bert_train.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Dataset

Dataset used:

Fake and Real News Dataset from Kaggle

🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset contains:
- Fake news articles
- Real news articles
- Labels for classification

---

# ⚙️ NLP Pipeline

The project follows this NLP pipeline:

```text
News Article
     ↓
Text Cleaning
     ↓
Lowercasing
     ↓
Remove Punctuation
     ↓
Remove Stopwords
     ↓
TF-IDF Vectorization
     ↓
Machine Learning / BERT
     ↓
Prediction
```

---

# 🤖 Models Used

## 1️⃣ Logistic Regression
Used with TF-IDF features for fast and accurate predictions.

## 2️⃣ Naive Bayes
Efficient probabilistic model for text classification.

## 3️⃣ Random Forest
Ensemble model for classification.

## 4️⃣ BERT Transformer
Advanced deep learning model for contextual understanding of news articles.

---

# 📈 Model Accuracy

| Model | Accuracy |
|------|------|
| Naive Bayes | ~94% |
| Logistic Regression | ~98% |
| Random Forest | ~99% |
| BERT | ~99%+ |

---

# 🖥️ Streamlit Web App

The application provides:

- ML Model Prediction
- BERT Prediction
- Confidence Score
- Interactive User Interface
- Sample News Examples

---

# 📸 Screenshots

## Home Page

```bash
https://drive.google.com/file/d/1q69A_Yq3BUCYrvijlpPgrd_SeEwdrZ6V/view?usp=drive_link
```

## Prediction Result

```bash
https://drive.google.com/file/d/1OMSzH-zceg8i73phZXPxHILwkZZXZ-s9/view?usp=drive_link
https://drive.google.com/file/d/1ExVugv-uNacHnPj_i6OZM4XF-flNwEp4/view?usp=drive_link
```

---

# ▶️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/ShreyaMaurya24/Fake_News_Detector.git
```

---

## 2️⃣ Navigate to Project Folder

```bash
cd FAKE_NEWS_DETECTION
```

---

## 3️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\\Scripts\\activate
```

---

## 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Run Application

## Run Streamlit App

```bash
streamlit run app.py
```

---

# 🧠 Train Machine Learning Model

```bash
python train_model.py
```

---

# 🤖 Train BERT Model

```bash
python bert_train.py
```

---

# 🔍 Example Predictions

## REAL NEWS

```text
Government launches new healthcare scheme for rural hospitals.
```

## FAKE NEWS

```text
Aliens secretly signed a peace treaty with world leaders.
```

---

# 🌟 Future Improvements

- Real-time News API Integration
- Explainable AI (SHAP/LIME)
- News URL Analysis
- Multilingual Fake News Detection
- RoBERTa / DistilBERT Models
- User Authentication
- Prediction History Storage

---

# 🎯 Learning Outcomes

Through this project, I learned:

- NLP preprocessing techniques
- Feature extraction using TF-IDF
- Machine Learning model training
- Transformer models (BERT)
- Streamlit web development
- Model deployment
- GitHub project management

---

# 👩‍💻 Author

Shreya Maurya
  
GitHub: https://github.com/ShreyaMaurya24

---

# ⭐ If you like this project

Give this repository a ⭐ on GitHub!