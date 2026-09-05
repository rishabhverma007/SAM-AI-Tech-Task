"""
TASK 1 : Fake News Detection
============================
Production-grade pipeline to classify news articles as REAL or FAKE.

Pipeline (as per the task sheet):
  1. Build a model to classify news articles as Real or Fake.
  2. Preprocess text using NLP techniques.
  3. Convert text into numerical features using TF-IDF.
  4. Train classification models such as Naive Bayes or Logistic Regression.
  5. Evaluate the model using Accuracy and F1-Score.

Extras that take this project beyond a basic submission:
  - 5-fold stratified cross-validation
  - Hyperparameter tuning with GridSearchCV
  - ROC-AUC evaluation
  - Confusion matrix, ROC curve and feature-importance plots (saved to output/)
  - Soft-voting ensemble of the tuned models
  - Best model + vectorizer exported with joblib for reuse

Usage:
    python fake_news_detection.py
    python fake_news_detection.py --data data/news.csv --test-size 0.2
    python fake_news_detection.py --predict "Your headline here"
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
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB

RANDOM_STATE = 42
OUTPUT_DIR = "output"
RESULTS_FILE = os.path.join(OUTPUT_DIR, "results.json")


# --------------------------------------------------------------------------
# 1. NLP text preprocessing
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, strip noise, remove stop words, and stem."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # remove URLs
    text = re.sub(r"<[^>]+>", " ", text)                    # remove HTML tags
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
    df = pd.read_csv(path)
    df = df[["title", "text", "label"]].dropna()
    df["content"] = (df["title"] + " " + df["text"]).apply(clean_text)
    df["label"] = df["label"].str.upper().map({"REAL": 1, "FAKE": 0})
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    print(f"[DATA] Loaded {len(df):,} articles "
          f"({df['label'].value_counts().to_dict()})")
    return df


# --------------------------------------------------------------------------
# 3. Plotting helpers
# --------------------------------------------------------------------------
def plot_confusion_matrix(y_true, y_pred, name: str, path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["FAKE", "REAL"])
    ax.set_yticks([0, 1], ["FAKE", "REAL"])
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
        # Simple ROC curve from sorted scores
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Fake News Detection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_top_features(vectorizer, model, path: str, n: int = 12):
    """Show the terms that most strongly indicate REAL vs FAKE news."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_.ravel()
    order = np.argsort(coefs)
    top_fake = feature_names[order[:n]][::-1]       # most negative -> FAKE
    top_real = feature_names[order[-n:]]            # most positive -> REAL

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, words, title, color in (
        (axes[0], top_real, "Strongest REAL indicators", "green"),
        (axes[1], top_fake, "Strongest FAKE indicators", "red"),
    ):
        ax.barh(words, [1] * len(words), color=color, alpha=0.75)
        ax.set_title(title)
        ax.set_xlabel("Weight (sign)")  # qualitative view
    fig.suptitle("Logistic Regression — Top Predictive Terms")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. Tuning + evaluation
# --------------------------------------------------------------------------
def tune_and_evaluate(name: str, estimator, param_grid, X_train, X_test,
                      y_train, y_test, vectorizer, results: dict):
    """Grid-search CV, then evaluate the best model on the hold-out set."""
    print(f"\n[TUNE] {name} — searching {param_grid}")
    grid = GridSearchCV(estimator, param_grid, scoring="f1", cv=5,
                        n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
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
    parser = argparse.ArgumentParser(description="Fake News Detection pipeline")
    parser.add_argument("--data", default="data/news.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--predict", help="classify a single headline + article")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1-2. Load + preprocess ----------------------------------------
    df = load_data(args.data)

    # ---- 3. TF-IDF feature extraction ----------------------------------
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                 sublinear_tf=True)
    X = vectorizer.fit_transform(df["content"])
    y = df["label"].values
    print(f"[FEATURES] TF-IDF matrix: {X.shape[0]:,} docs x "
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
        X_train, X_test, y_train, y_test, vectorizer, results)

    models["Logistic Regression"] = tune_and_evaluate(
        "Logistic Regression", LogisticRegression(max_iter=2000,
                                                  random_state=RANDOM_STATE),
        {"C": [0.1, 1.0, 10.0]},
        X_train, X_test, y_train, y_test, vectorizer, results)

    # ---- Ensemble: soft-voting of the tuned models ---------------------
    ensemble = VotingClassifier(
        estimators=[("nb", models["Naive Bayes"]),
                    ("lr", models["Logistic Regression"])],
        voting="soft")
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    results["Ensemble (NB+LR)"] = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(
            y_test, ensemble.predict_proba(X_test)[:, 1]), 4),
    }
    print(f"\n[ENSEMBLE] Soft-voting Naive Bayes + Logistic Regression")
    print(f"  Test Accuracy : {results['Ensemble (NB+LR)']['accuracy']:.4f}")
    print(f"  Test F1-Score : {results['Ensemble (NB+LR)']['f1']:.4f}")
    print(f"  Test ROC-AUC  : {results['Ensemble (NB+LR)']['roc_auc']:.4f}")
    print("\n" + classification_report(y_test, y_pred,
                                       target_names=["FAKE", "REAL"]))
    models["Ensemble"] = ensemble

    # ---- 5. Plots + export ---------------------------------------------
    plot_roc_curves(models, X_test, y_test,
                    os.path.join(OUTPUT_DIR, "roc_curves.png"))
    plot_top_features(vectorizer, models["Logistic Regression"],
                      os.path.join(OUTPUT_DIR, "top_features.png"))

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EXPORT] Metrics  -> {RESULTS_FILE}")
    print(f"[EXPORT] Plots    -> {OUTPUT_DIR}/")
    joblib.dump(ensemble, os.path.join(OUTPUT_DIR, "fake_news_model.joblib"))
    joblib.dump(vectorizer, os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib"))
    print(f"[EXPORT] Model    -> {OUTPUT_DIR}/fake_news_model.joblib")

    # ---- Optional: single prediction -----------------------------------
    if args.predict:
        content = clean_text(args.predict)
        proba = ensemble.predict_proba(vectorizer.transform([content]))[0]
        label = "REAL" if proba[1] >= 0.5 else "FAKE"
        print(f"\n[PREDICT] \"{args.predict[:80]}...\"")
        print(f"          -> {label} "
              f"(confidence: {max(proba):.1%})")


if __name__ == "__main__":
    main()