# ================================
# FILE: train_model.py
# ================================

import pandas as pd
import nltk
import re
import pickle

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# DOWNLOAD STOPWORDS
# --------------------------------
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# --------------------------------
# LOAD DATASETS
# --------------------------------
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

print("Fake shape:", fake.shape)
print("True shape:", true.shape)

# --------------------------------
# ADD LABELS
# Fake = 0
# Real = 1
# --------------------------------
fake["label"] = 0
true["label"] = 1

# --------------------------------
# COMBINE DATA
# --------------------------------
data = pd.concat([fake, true])

# Shuffle
data = data.sample(frac=1, random_state=42)

# Keep important columns
data = data[["text", "label"]]

# Remove missing values
data = data.dropna()

print("\nDataset Loaded Successfully")
print(data.head())

# --------------------------------
# TEXT CLEANING FUNCTION
# --------------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
data["text"] = data["text"].apply(clean_text)

print("\nText Cleaning Completed")

# --------------------------------
# SAVE CLEAN DATA
# --------------------------------
data.to_csv("dataset/clean_news.csv", index=False)

# --------------------------------
# FEATURE EXTRACTION (TF-IDF)
# --------------------------------
X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(X)

print("\nTF-IDF Shape:", X.shape)

# --------------------------------
# TRAIN TEST SPLIT
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)

# =========================================
# LOGISTIC REGRESSION
# =========================================
print("\nTraining Logistic Regression...")

lr_model = LogisticRegression(max_iter=2000)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_accuracy)

# =========================================
# NAIVE BAYES
# =========================================
print("\nTraining Naive Bayes...")

nb_model = MultinomialNB()

nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)

nb_accuracy = accuracy_score(y_test, nb_pred)

print("\nNaive Bayes Accuracy:", nb_accuracy)

# =========================================
# RANDOM FOREST
# =========================================
print("\nTraining Random Forest...")

rf_model = RandomForestClassifier(n_estimators=100)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)

# =========================================
# MODEL EVALUATION
# =========================================
print("\nClassification Report:\n")

print(classification_report(y_test, lr_pred))

print("\nConfusion Matrix:\n")

print(confusion_matrix(y_test, lr_pred))

# =========================================
# SAVE MODEL
# =========================================
with open("models/fake_news_model.pkl", "wb") as f:
    pickle.dump(lr_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel Saved Successfully!")