"""
TASK 2 : Email Spam Classifier
==============================
Production-grade pipeline to detect spam emails and display the predicted
category.

Pipeline (as per the task sheet):
  1. Develop a model to detect spam emails.
  2. Clean and preprocess email text.
  3. Apply TF-IDF vectorization.
  4. Train a classification model (Naive Bayes / SVM).
  5. Display the predicted email category.

Extras that take this project beyond a basic submission:
  - 5-fold stratified cross-validation
  - Hyperparameter tuning with GridSearchCV
  - SVM calibrated with CalibratedClassifierCV (real probabilities)
  - ROC-AUC evaluation
  - Confusion matrix, ROC curve and spam-term plots (saved to output/)
  - Soft-voting ensemble of the tuned models
  - Best model + vectorizer exported with joblib for reuse

Usage:
    python email_spam_classifier.py
    python email_spam_classifier.py --data data/spam.csv
    python email_spam_classifier.py --predict "Your email text here"
"""

import argparse
import json
import os
import re

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

RANDOM_STATE = 42
OUTPUT_DIR = "output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.json")


# --------------------------------------------------------------------------
# 1. NLP text preprocessing
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# 2. Data loading
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]
    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    df["clean_message"] = df["message"].apply(clean_text)
    print(f"[DATA] Loaded {len(df):,} messages "
          f"({df['label'].value_counts().to_dict()})")
    return df


# --------------------------------------------------------------------------
# 3. Plotting helpers
# --------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, name: str, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["HAM", "SPAM"])
    ax.set_yticks([0, 1], ["HAM", "SPAM"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{name} — Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(models: dict, X_test, y_test, path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Email Spam Classification")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spam_terms(vectorizer, svm_model, path: str, n: int = 12):
    """Show the terms that most strongly indicate SPAM (SVM coefficients)."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = svm_model.coef_.ravel()
    order = np.argsort(coefs)
    top_spam = feature_names[order[-n:]][::-1]   # most positive -> SPAM

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top_spam, [1] * n, color="crimson", alpha=0.8)
    ax.set_title("SVM — Top Terms That Trigger SPAM Detection")
    ax.set_xlabel("Weight (sign)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Tuning + evaluation
# --------------------------------------------------------------------------
def tune_and_evaluate(name: str, estimator, param_grid, X_train, X_test,
                      y_train, y_test, results: dict,
                      calibrate: bool = False):
    """Grid-search CV, optionally calibrate, then evaluate on the hold-out set."""
    print(f"\n[TUNE] {name} — searching {param_grid}")
    grid = GridSearchCV(estimator, param_grid, scoring="f1", cv=5,
                        n_jobs=-1)
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    if calibrate:  # SVM: turn decision scores into real probabilities
        best = CalibratedClassifierCV(best, cv=3)
        best.fit(X_train, y_train)

    y_pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]

    metrics = {
        "best_params": grid.best_params_,
        "cv_mean_f1": round(grid.best_score_, 4),
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }
    results[name] = metrics

    print(f"  Best params        : {grid.best_params_}")
    print(f"  5-fold CV F1       : {metrics['cv_mean_f1']:.4f}")
    print(f"  Test Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  Test Precision     : {metrics['precision']:.4f}")
    print(f"  Test Recall        : {metrics['recall']:.4f}")
    print(f"  Test F1-Score      : {metrics['f1']:.4f}")
    print(f"  Test ROC-AUC       : {metrics['roc_auc']:.4f}")

    plot_confusion_matrix(y_test, y_pred, name,
                          os.path.join(OUTPUT_DIR, f"confusion_matrix_{name}.png"))
    return best


def main():
    parser = argparse.ArgumentParser(description="Email Spam Classifier pipeline")
    parser.add_argument("--data", default="data/spam.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--predict", help="classify a single email/message")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1-2. Load + preprocess ----------------------------------------
    df = load_data(args.data)

    # ---- 3. TF-IDF vectorization ---------------------------------------
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                 sublinear_tf=True)
    X = vectorizer.fit_transform(df["clean_message"])
    y = df["label"].values
    print(f"[FEATURES] TF-IDF matrix: {X.shape[0]:,} messages x "
          f"{X.shape[1]:,} terms (unigrams + bigrams)")

    # ---- Train / test split --------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=RANDOM_STATE,
        stratify=y)
    print(f"[SPLIT] Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} "
          f"(stratified)")

    # ---- 4. Tune + train models ----------------------------------------
    results = {}
    models = {}

    models["Naive Bayes"] = tune_and_evaluate(
        "Naive Bayes", MultinomialNB(),
        {"alpha": [0.1, 0.5, 1.0]},
        X_train, X_test, y_train, y_test, results)

    models["SVM"] = tune_and_evaluate(
        "SVM", LinearSVC(random_state=RANDOM_STATE),
        {"C": [0.1, 1.0, 10.0]},
        X_train, X_test, y_train, y_test, results, calibrate=True)

    # ---- Ensemble: soft-voting of the tuned models ---------------------
    ensemble = VotingClassifier(
        estimators=[("nb", models["Naive Bayes"]),
                    ("svm", models["SVM"])],
        voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    results["Ensemble (NB+SVM)"] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(
            y_test, ensemble.predict_proba(X_test)[:, 1]), 4),
    }
    print(f"\n[ENSEMBLE] Soft-voting Naive Bayes + SVM")
    print(f"  Test Accuracy : {results['Ensemble (NB+SVM)']['accuracy']:.4f}")
    print(f"  Test F1-Score : {results['Ensemble (NB+SVM)']['f1']:.4f}")
    print(f"  Test ROC-AUC  : {results['Ensemble (NB+SVM)']['roc_auc']:.4f}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["HAM", "SPAM"]))
    models["Ensemble"] = ensemble

    # ---- 5. Plots + export ---------------------------------------------
    plot_roc_curves(models, X_test, y_test,
                    os.path.join(OUTPUT_DIR, "roc_curves.png"))
    # The calibrated SVM wraps the tuned LinearSVC in `.estimator`
    plot_spam_terms(vectorizer, models["SVM"].estimator,
                    os.path.join(OUTPUT_DIR, "spam_terms.png"))

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXPORT] Metrics  -> {RESULTS_FILE}")
    print(f"[EXPORT] Plots    -> {OUTPUT_DIR}/")
    joblib.dump(ensemble, os.path.join(OUTPUT_DIR, "spam_model.joblib"))
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
    print(f"[EXPORT] Model    -> {OUTPUT_DIR}/spam_model.joblib")

    # ---- Optional: single prediction -----------------------------------
    if args.predict:
        content = clean_text(args.predict)
        proba = ensemble.predict_proba(vectorizer.transform([content]))[0]
        label = "SPAM" if proba[1] >= 0.5 else "HAM"
        print(f"\n[PREDICT] \"{args.predict[:80]}...\"")
        print(f"          -> {label} (confidence: {max(proba):.1%})")


if __name__ == "__main__":
    main()