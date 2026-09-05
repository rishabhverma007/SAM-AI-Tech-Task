"""
TASK 3 : Text Summarization
===========================
Build a system to generate summaries from long articles.

Pipeline (as per the task sheet):
  1. Build a system to generate summaries from long articles.
  2. Clean and preprocess text data.
  3. Apply extractive text summarization techniques.
  4. Compare original and summarized text.
  5. Evaluate summary quality.

Two extractive techniques are implemented and compared:
  - Frequency-based scoring (word frequency weighted by sentence length)
  - TF-IDF-based scoring (sentence importance from a TF-IDF matrix)

Usage:
    python text_summarization.py                       # uses default article
    python text_summarization.py --file path/to.txt    # any text file
    python text_summarization.py --top_n 5             # sentences in summary
"""

import argparse
import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DEFAULT_ARTICLE = "sample_articles/example_article.txt"


# --------------------------------------------------------------------------
# 1. Load + preprocess
# --------------------------------------------------------------------------
def load_article(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Lowercase and strip noise, keeping sentence structure intact."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list:
    """Split text into sentences on sentence-ending punctuation."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.split()) > 3]


def tokenize(sentence: str) -> list:
    """Word tokens minus stop words."""
    return [
        w for w in re.findall(r"[a-z0-9']+", sentence)
        if w not in ENGLISH_STOP_WORDS and len(w) > 1
    ]


# --------------------------------------------------------------------------
# 2. Extractive summarization techniques
# --------------------------------------------------------------------------
def frequency_scores(sentences: list) -> dict:
    """Score sentences by normalized word frequency + position bonus."""
    word_freq = Counter()
    for s in sentences:
        word_freq.update(tokenize(s))
    if not word_freq:
        return {}
    max_freq = max(word_freq.values())
    # Normalize so common words (like 'said') don't dominate everything
    freq = {w: f / max_freq for w, f in word_freq.items()}

    scores = {}
    n = len(sentences)
    for i, s in enumerate(sentences):
        words = tokenize(s)
        if not words:
            continue
        score = sum(freq.get(w, 0) for w in words) / len(words)
        # Sentences at the start of an article usually carry key info
        score += 0.15 * (1 - i / n)
        scores[s] = score
    return scores


def tfidf_scores(sentences: list) -> dict:
    """Score sentences using TF-IDF: sum of TF-IDF weights of its words."""
    cleaned = [" ".join(tokenize(s)) for s in sentences]
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(cleaned)
    sums = matrix.sum(axis=1).A1  # per-sentence total TF-IDF weight
    # Normalize by sentence length so long sentences aren't favoured blindly
    lengths = [max(len(tokenize(s)), 1) for s in sentences]
    return {s: (sums[i] / lengths[i]) for i, s in enumerate(sentences)}


def summarize(sentences: list, scores: dict, top_n: int = 5) -> list:
    """Pick the top-N highest-scoring sentences, preserving original order."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = {s for s, _ in ranked[:top_n]}
    return [s for s in sentences if s in top]


# --------------------------------------------------------------------------
# 3. Compare original vs summarized text
# --------------------------------------------------------------------------
def compare(original: str, summary: list, top_keywords: list) -> dict:
    """Word/sentence stats comparing the article with its summary."""
    orig_words = len(original.split())
    summ_words = len(" ".join(summary).split())
    orig_sentences = len(split_sentences(original))
    summ_tokens = set(tokenize(" ".join(summary)))
    keywords_kept = sum(1 for k in top_keywords if k in summ_tokens)

    return {
        "original_words": orig_words,
        "summary_words": summ_words,
        "compression_ratio": f"{summ_words / orig_words:.1%}",
        "original_sentences": orig_sentences,
        "summary_sentences": len(summary),
        "keywords_kept": f"{keywords_kept}/{len(top_keywords)}",
    }


# --------------------------------------------------------------------------
# 4. Evaluate summary quality
# --------------------------------------------------------------------------
def rouge1(original_sentences: list, summary: list) -> dict:
    """ROUGE-1 (unigram overlap) between summary and source text."""
    orig_tokens = Counter()
    for s in original_sentences:
        orig_tokens.update(tokenize(s))
    summ_tokens = Counter()
    for s in summary:
        summ_tokens.update(tokenize(s))

    overlap = sum((orig_tokens & summ_tokens).values())
    total_summary = sum(summ_tokens.values())
    total_orig = sum(orig_tokens.values())

    precision = overlap / total_summary if total_summary else 0.0
    recall = overlap / total_orig if total_orig else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": f"{precision:.2%}", "recall": f"{recall:.2%}",
            "f1": f"{f1:.2%}"}


def top_keywords(sentences: list, k: int = 10) -> list:
    freq = Counter()
    for s in sentences:
        freq.update(tokenize(s))
    return [w for w, _ in freq.most_common(k)]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extractive text summarizer")
    parser.add_argument("--file", default=DEFAULT_ARTICLE,
                        help="path to the article to summarize")
    parser.add_argument("--top_n", type=int, default=5,
                        help="number of sentences in the summary")
    args = parser.parse_args()

    # 1. Load + preprocess
    raw = load_article(args.file)
    text = clean_text(raw)
    sentences = split_sentences(text)
    print(f"Loaded article: {args.file}")
    print(f"Article length : {len(text.split())} words, "
          f"{len(sentences)} sentences")

    # 2. Apply extractive techniques
    freq_summary = summarize(sentences, frequency_scores(sentences), args.top_n)
    tfidf_summary = summarize(sentences, tfidf_scores(sentences), args.top_n)

    kw = top_keywords(sentences)

    for name, summary in (("Frequency-based", freq_summary),
                          ("TF-IDF-based  ", tfidf_summary)):
        print(f"\n{'=' * 60}\n{name} summary "
              f"({len(summary)} sentences)\n{'=' * 60}")
        print(" ".join(summary))

        # 4. Compare original vs summarized text
        stats = compare(text, summary, kw)
        print(f"\n  Original: {stats['original_words']} words / "
              f"{stats['original_sentences']} sentences")
        print(f"  Summary : {stats['summary_words']} words / "
              f"{stats['summary_sentences']} sentences "
              f"(compression to {stats['compression_ratio']})")
        print(f"  Top keywords kept: {stats['keywords_kept']}")

        # 5. Evaluate summary quality
        rouge = rouge1(sentences, summary)
        print(f"  ROUGE-1 -> precision {rouge['precision']} | "
              f"recall {rouge['recall']} | F1 {rouge['f1']}")


if __name__ == "__main__":
    main()