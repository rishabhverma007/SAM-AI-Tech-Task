"""
TASK 1 : Fake News Detection
============================
Build a model to classify news articles as REAL or FAKE.

Pipeline (as per the task sheet):
  1. Load a labelled dataset of news articles.
  2. Preprocess the text using NLP techniques
     (lowercasing, cleaning, stop-word removal, stemming).
  3. Convert the text into numerical features using TF-IDF.
  4. Train classification models: Naive Bayes and Logistic Regression.
  5. Evaluate the models using Accuracy and F1-Score.

Usage:
    python fake_news_detection.py
"""

import re

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = "data/news.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000


def clean_text(text: str) -> str:
    """NLP preprocessing: lowercase, remove noise, remove stop words, stem."""
    if not isinstance(text, str):
        return ""
    # Lowercase and strip HTML/URL artefacts
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"<[^>]+>", " ", text)                    # remove HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                   # keep letters only
    # Tokenize, drop stop words, then stem each word
    stemmer = PorterStemmer()
    words = [
        stemmer.stem(w)
        for w in text.split()
        if w not in ENGLISH_STOP_WORDS and len(w) > 1
    ]
    return " ".join(words)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load and prepare the news dataset."""
    df = pd.read_csv(path)
    df = df[["title", "text", "label"]].dropna()
    # Combine title + body so the model sees the full article
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)
    df["label"] = df["label"].str.upper().map({"REAL": 1, "FAKE": 0})
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    print(f"Loaded {len(df)} articles "
          f"({df['label'].value_counts().to_dict()})")
    return df


def train_and_evaluate(model, name: str, X_train, X_test, y_train, y_test):
    """Train a classifier and report Accuracy + F1-Score."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n===================== {name} =====================")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-Score : {f1_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    return model


def predict_article(model, vectorizer, title: str, text: str) -> str:
    """Classify a single article and print the predicted category."""
    content = clean_text(f"{title} {text}")
    proba = model.predict_proba(vectorizer.transform([content]))[0]
    label = "REAL" if model.predict(vectorizer.transform([content]))[0] == 1 else "FAKE"
    return f"{label} (confidence: {max(proba):.1%})"


def main():
    df = load_data()

    # ---- Step 3: TF-IDF feature extraction -------------------------------
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["content"])
    y = df["label"]

    # ---- Train / test split ----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} articles | Test: {X_test.shape[0]} articles")

    # ---- Step 4: Train Naive Bayes + Logistic Regression -----------------
    nb_model = train_and_evaluate(
        MultinomialNB(), "Naive Bayes", X_train, X_test, y_train, y_test
    )
    lr_model = train_and_evaluate(
        LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Logistic Regression",
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ---- Demo: predict a couple of unseen headlines ---------------------
    print("\n====================== Demo predictions =======================")
    demos = [
        ("Breaking: Scientists discover new energy source",
         "Researchers at a national laboratory announced a breakthrough in "
         "clean energy production during a press conference this morning."),
        ("You won't believe what this celebrity did!",
         "SHOCKING! Doctors hate this one trick that will make you rich "
         "overnight. Click now to claim your prize!"),
    ]
    for title, body in demos:
        pred_nb = predict_article(nb_model, vectorizer, title, body)
        pred_lr = predict_article(lr_model, vectorizer, title, body)
        print(f"- \"{title}\"\n    Naive Bayes       -> {pred_nb}\n"
              f"    Logistic Reg.     -> {pred_lr}")


if __name__ == "__main__":
    main()