# Task 1 — Fake News Detection

Classify news articles as **REAL** or **FAKE** using NLP and machine learning.

## Task Requirements (from the task sheet)

- ✅ Build a model to classify news articles as Real or Fake.
- ✅ Preprocess text using NLP techniques.
- ✅ Convert text into numerical features using TF-IDF.
- ✅ Train classification models such as Naive Bayes or Logistic Regression.
- ✅ Evaluate the model using Accuracy and F1-Score.

## Project Structure

```
Task-1-Fake-News-Detection/
├── fake_news_detection.py   # main script
├── requirements.txt
├── data/
│   └── news.csv             # labelled dataset (REAL / FAKE articles)
└── README.md
```

## Dataset

`data/news.csv` is the public **fake_or_real_news** dataset (title, text, label).
Each row is a news article labelled `REAL` or `FAKE`.

## How to Run

```bash
pip install -r requirements.txt
python fake_news_detection.py
```

## How It Works

| Step | Technique |
|------|-----------|
| 1. Text cleaning | Lowercase, strip URLs / HTML / punctuation |
| 2. NLP preprocessing | Stop-word removal + Porter stemming |
| 3. Feature extraction | TF-IDF vectorization (unigrams + bigrams) |
| 4. Models | Multinomial Naive Bayes + Logistic Regression |
| 5. Evaluation | Accuracy, F1-Score, classification report, confusion matrix |