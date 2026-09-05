# Task 2 — Email Spam Classifier

Detect **spam** vs **ham** (legitimate) emails and display the predicted category.

## Task Requirements (from the task sheet)

- ✅ Develop a model to detect spam emails.
- ✅ Clean and preprocess email text.
- ✅ Apply TF-IDF vectorization.
- ✅ Train a classification model (Naive Bayes / SVM).
- ✅ Display the predicted email category.

## Project Structure

```
Task-2-Email-Spam-Classifier/
├── email_spam_classifier.py   # main script
├── requirements.txt
├── data/
│   └── spam.csv               # SMS Spam Collection dataset
└── README.md
```

## Dataset

`data/spam.csv` is the public **SMS Spam Collection** (5,572 messages)
labelled `ham` / `spam`.

## How to Run

```bash
pip install -r requirements.txt
python email_spam_classifier.py
```

## How It Works

| Step | Technique |
|------|-----------|
| 1. Text cleaning | Lowercase, strip URLs, numbers, punctuation |
| 2. NLP preprocessing | Stop-word removal + Porter stemming |
| 3. Feature extraction | TF-IDF vectorization (unigrams + bigrams) |
| 4. Models | Multinomial Naive Bayes + Linear SVM |
| 5. Prediction | Displays predicted category (HAM / SPAM) with confidence |