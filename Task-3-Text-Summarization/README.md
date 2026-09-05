# Task 3 — Text Summarization

Generate short, meaningful summaries from long articles using **extractive**
text summarization.

## Task Requirements (from the task sheet)

- ✅ Build a system to generate summaries from long articles.
- ✅ Clean and preprocess text data.
- ✅ Apply extractive text summarization techniques.
- ✅ Compare original and summarized text.
- ✅ Evaluate summary quality.

## Project Structure

```
Task-3-Text-Summarization/
├── text_summarization.py        # main script
├── requirements.txt
├── sample_articles/
│   └── example_article.txt      # long article used for the demo
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt

python text_summarization.py                          # default article
python text_summarization.py --file path/to/article.txt   # any text file
python text_summarization.py --top_n 5                # summary sentence count
```

## How It Works

Two extractive techniques are implemented and compared:

1. **Frequency-based scoring** — sentences are scored by the normalized
   frequency of their words (with a small bonus for early sentences).
2. **TF-IDF-based scoring** — sentences are scored by the summed TF-IDF
   weight of their words, normalized by sentence length.

The top-N sentences are picked and re-ordered to read naturally.

## Evaluation

Summary quality is evaluated with:

- **Compression ratio** — summary size vs. original size.
- **Keyword coverage** — how many of the article's top keywords appear in
  the summary.
- **ROUGE-1** — unigram overlap between the summary and the source text
  (precision, recall, F1), a standard metric for summarization quality.