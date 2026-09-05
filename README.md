# SAM AI Tech Task

Machine Learning internship tasks completed for **SAM AI Technologies**.

This repository contains one self-contained project per task, organised
task-wise. Each project includes its own README, requirements, dataset, and
a runnable Python script.

## 📋 Task List

| # | Task | Project | Status |
|---|------|---------|--------|
| 1 | **Fake News Detection** — classify news articles as Real or Fake using TF-IDF + Naive Bayes / Logistic Regression | [`Task-1-Fake-News-Detection/`](Task-1-Fake-News-Detection/) | ✅ |
| 2 | **Email Spam Classifier** — detect spam emails using TF-IDF + Naive Bayes / SVM | [`Task-2-Email-Spam-Classifier/`](Task-2-Email-Spam-Classifier/) | ✅ |
| 3 | **Text Summarization** — extractive summarization of long articles | [`Task-3-Text-Summarization/`](Task-3-Text-Summarization/) | ✅ |

> Additional tasks will be added here as they are assigned.

## 🛠 Tech Stack

- **Python 3.11**
- **scikit-learn** — TF-IDF vectorization, Naive Bayes, Logistic Regression, SVM
- **NLTK** — Porter stemming for text preprocessing
- **pandas** — data loading and manipulation

## 🚀 How to Run

Each task is independent. From the task folder:

```bash
pip install -r requirements.txt
python <task_script>.py
```

| Task | Folder | Script |
|------|--------|--------|
| 1 | `Task-1-Fake-News-Detection/` | `fake_news_detection.py` |
| 2 | `Task-2-Email-Spam-Classifier/` | `email_spam_classifier.py` |
| 3 | `Task-3-Text-Summarization/` | `text_summarization.py` |

## 📁 Repository Structure

```
SAM-AI-Tech-Task/
├── README.md
├── Task-1-Fake-News-Detection/
│   ├── fake_news_detection.py
│   ├── requirements.txt
│   └── data/news.csv
├── Task-2-Email-Spam-Classifier/
│   ├── email_spam_classifier.py
│   ├── requirements.txt
│   └── data/spam.csv
└── Task-3-Text-Summarization/
    ├── text_summarization.py
    ├── requirements.txt
    └── sample_articles/example_article.txt
```

## 📌 Submission

Repo: **https://github.com/rishabhverma007/SAM-AI-Tech-Task**

— Rishabh Verma