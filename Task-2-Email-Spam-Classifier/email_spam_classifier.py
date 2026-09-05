"""
TASK 2 : Email Spam Classifier
==============================
Develop a model to detect spam emails and display the predicted category.

Pipeline (as per the task sheet):
  1. Develop a model to detect spam emails.
  2. Clean and preprocess email text.
  3. Apply TF-IDF vectorization.
  4. Train a classification model (Naive Bayes / SVM).
  5. Display the predicted email category.

Usage:
    python email_spam_classifier.py
"""

import re

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

DATA_PATH = "data/spam.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 10000


def clean_text(text: str) -> str:
    """Clean and preprocess email/message text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"\b\d+\b", " ", text)                    # remove numbers
    text = re.sub(r"[^a-z\s]", " ", text)                   # keep letters only
    stemmer = PorterStemmer()
    words = [
        stemmer.stem(w)
        for w in text.split()
        if w not in ENGLISH_STOP_WORDS and len(w) > 1
    ]
    return " ".join(words)


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the SMS Spam Collection and prepare it for training."""
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]
    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    df["clean_message"] = df["message"].apply(clean_text)
    print(f"Loaded {len(df)} messages "
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
    print(classification_report(y_test, y_pred, target_names=["HAM", "SPAM"]))
    print("Confusion matrix (rows=true, cols=predicted):")
    print(confusion_matrix(y_test, y_pred))
    return model


def predict_category(model, vectorizer, text: str) -> str:
    """Classify one email and display its predicted category."""
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    label = "SPAM" if model.predict(vec)[0] == 1 else "HAM"

    # Naive Bayes exposes probabilities; SVM exposes a decision score.
    # Convert the SVM decision score to a pseudo-probability via sigmoid.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]
        confidence = max(proba)
    else:
        import math
        score = model.decision_function(vec)[0]
        confidence = 1.0 / (1.0 + math.exp(-score))

    return f"{label} (confidence: {confidence:.1%})"


def main():
    df = load_data()

    # ---- Step 3: TF-IDF vectorization ------------------------------------
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["clean_message"])
    y = df["label"]

    # ---- Train / test split ----------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {X_train.shape[0]} messages | Test: {X_test.shape[0]} messages")

    # ---- Step 4: Train Naive Bayes + SVM --------------------------------
    nb_model = train_and_evaluate(
        MultinomialNB(), "Naive Bayes", X_train, X_test, y_train, y_test
    )
    svm_model = train_and_evaluate(
        LinearSVC(random_state=RANDOM_STATE),
        "SVM (Linear SVC)",
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # ---- Step 5: Display predicted category for new emails ---------------
    print("\n================= Predict new emails ===================")
    new_emails = [
        "Dear customer, your account has been suspended. "
        "Verify immediately at our secure portal to avoid closure.",
        "Hi Rishabh, the meeting has been moved to 3 PM tomorrow. "
        "Please review the attached report before we start.",
        "CONGRATULATIONS! You have won a FREE iPhone. "
        "Claim your prize now by sending your bank details!",
        "Reminder: your project submission is due Friday. "
        "Let me know if you need any help with the write-up.",
    ]
    for msg in new_emails:
        pred_nb = predict_category(nb_model, vectorizer, msg)
        pred_svm = predict_category(svm_model, vectorizer, msg)
        print(f"- \"{msg[:60]}...\"\n    Naive Bayes -> {pred_nb}\n"
              f"    SVM         -> {pred_svm}")


if __name__ == "__main__":
    main()