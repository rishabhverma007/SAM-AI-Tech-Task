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

Three extractive techniques are implemented and compared:
  1. Frequency-based scoring  — word frequency + position bonus
  2. TF-IDF-based scoring     — summed TF-IDF weights per sentence
  3. TextRank                 — graph-based PageRank over sentence similarity

Evaluation:
  - Compression ratio + keyword coverage
  - ROUGE-1 / ROUGE-2 against a gold reference summary
  - Bar-chart comparison of all methods (saved to output/)

Usage:
    python text_summarization.py
    python text_summarization.py --file path/to.txt --top_n 5
"""

import argparse
import os
import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")  # headless-safe plotting
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DEFAULT_ARTICLE = "sample_articles/example_article.txt"
DEFAULT_REFERENCE = "sample_articles/example_article_reference_summary.txt"
OUTPUT_DIR = "output"


# --------------------------------------------------------------------------
# 1. Load + preprocess
# --------------------------------------------------------------------------
def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Lowercase and strip noise, keeping sentence structure intact."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s.!?']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if len(s.split()) > 3]


def tokenize(sentence: str) -> list:
    """Word tokens minus stop words."""
    return [
        w for w in re.findall(r"[a-z0-9']+", sentence)
        if w not in ENGLISH_STOP_WORDS and len(w) > 1
    ]


# --------------------------------------------------------------------------
# 2. Extractive scoring techniques
# --------------------------------------------------------------------------
def frequency_scores(sentences: list) -> dict:
    """Score sentences by normalized word frequency + position bonus."""
    word_freq = Counter()
    for s in sentences:
        word_freq.update(tokenize(s))
    if not word_freq:
        return {}
    max_freq = max(word_freq.values())
    freq = {w: f / max_freq for w, f in word_freq.items()}

    scores, n = {}, len(sentences)
    for i, s in enumerate(sentences):
        words = tokenize(s)
        if not words:
            continue
        score = sum(freq.get(w, 0) for w in words) / len(words)
        score += 0.15 * (1 - i / n)  # early sentences carry key info
        scores[s] = score
    return scores


def tfidf_scores(sentences: list) -> dict:
    """Score sentences by summed TF-IDF weight, normalized by length."""
    cleaned = [" ".join(tokenize(s)) for s in sentences]
    matrix = TfidfVectorizer().fit_transform(cleaned)
    sums = matrix.sum(axis=1).A1
    lengths = [max(len(tokenize(s)), 1) for s in sentences]
    return {s: sums[i] / lengths[i] for i, s in enumerate(sentences)}


def textrank_scores(sentences: list, damping: float = 0.85,
                    max_iter: int = 200, tol: float = 1e-4) -> dict:
    """Graph-based summarization: PageRank over sentence similarity."""
    n = len(sentences)
    tokens = [set(tokenize(s)) for s in sentences]

    # Sentence similarity graph (Jaccard overlap)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            union = tokens[i] | tokens[j]
            sim[i, j] = len(tokens[i] & tokens[j]) / len(union) if union else 0.0

    # Row-normalize -> transition matrix
    row_sums = sim.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    transition = sim / row_sums

    # Power-iteration PageRank
    scores = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(new_scores - scores).sum() < tol:
            scores = new_scores
            break
        scores = new_scores

    return {s: scores[i] for i, s in enumerate(sentences)}


def summarize(sentences: list, scores: dict, top_n: int = 5) -> list:
    """Pick the top-N sentences, preserving original order."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = {s for s, _ in ranked[:top_n]}
    return [s for s in sentences if s in top]


# --------------------------------------------------------------------------
# 3. Compare original vs summarized text
# --------------------------------------------------------------------------
def compare(original: str, summary: list, top_keywords: list) -> dict:
    orig_words = len(original.split())
    summ_words = len(" ".join(summary).split())
    orig_sentences = len(split_sentences(original))
    summ_tokens = set(tokenize(" ".join(summary)))
    keywords_kept = sum(1 for k in top_keywords if k in summ_tokens)

    return {
        "original_words": orig_words,
        "summary_words": summ_words,
        "compression_ratio": summ_words / orig_words,
        "original_sentences": orig_sentences,
        "summary_sentences": len(summary),
        "keywords_kept": f"{keywords_kept}/{len(top_keywords)}",
    }


def top_keywords(sentences: list, k: int = 10) -> list:
    freq = Counter()
    for s in sentences:
        freq.update(tokenize(s))
    return [w for w, _ in freq.most_common(k)]


# --------------------------------------------------------------------------
# 4. Evaluate summary quality (ROUGE against gold reference)
# --------------------------------------------------------------------------
def rouge_n(summary: list, reference: str, n: int = 1) -> dict:
    """ROUGE-N precision / recall / F1 between a summary and a reference."""
    def ngrams(tokens: list, n: int) -> Counter:
        return Counter(tuple(tokens[i:i + n])
                       for i in range(len(tokens) - n + 1))

    summ_tokens = tokenize(" ".join(summary))
    ref_tokens = tokenize(reference)
    if len(summ_tokens) < n or len(ref_tokens) < n:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    summ_grams, ref_grams = ngrams(summ_tokens, n), ngrams(ref_tokens, n)
    overlap = sum((summ_grams & ref_grams).values())
    precision = overlap / sum(summ_grams.values())
    recall = overlap / sum(ref_grams.values())
    f1 = (2 * precision * recall / (precision + recall)) \
        if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------
# 5. Plotting
# --------------------------------------------------------------------------
def plot_comparison(method_stats: dict, path: str):
    """Bar chart comparing methods on ROUGE-1 F1 and compression."""
    names = list(method_stats.keys())
    f1 = [method_stats[m]["rouge1_f1"] for m in names]
    comp = [method_stats[m]["compression"] for m in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - 0.2, f1, 0.4, label="ROUGE-1 F1",
                   color="teal", alpha=0.9)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + 0.2, comp, 0.4, label="Compression %",
                    color="orange", alpha=0.9)

    ax.set_xticks(x, names)
    ax.set_ylabel("ROUGE-1 F1")
    ax2.set_ylabel("Summary size (% of original)")
    ax.set_title("Summarization Method Comparison")
    ax.set_ylim(0, max(f1) * 1.3)
    ax2.set_ylim(0, max(comp) * 1.4)
    for b, v in zip(bars1, f1):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=9)
    for b, v in zip(bars2, comp):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                 ha="center", fontsize=9)
    fig.legend(loc="upper right", bbox_to_anchor=(0.92, 0.92))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extractive text summarizer")
    parser.add_argument("--file", default=DEFAULT_ARTICLE,
                        help="path to the article to summarize")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE,
                        help="path to a gold reference summary (for ROUGE)")
    parser.add_argument("--top_n", type=int, default=5,
                        help="number of sentences in the summary")
    args = parser.parse_args()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load + preprocess
    raw = load_file(args.file)
    text = clean_text(raw)
    sentences = split_sentences(text)
    reference = clean_text(load_file(args.reference)) \
        if os.path.exists(args.reference) else ""
    kw = top_keywords(sentences)

    print(f"[DATA] Loaded article   : {args.file}")
    print(f"[DATA] {len(text.split())} words, {len(sentences)} sentences")
    if reference:
        print(f"[DATA] Reference summary: {args.reference} "
              f"({len(reference.split())} words)")

    # 2. Apply the three extractive techniques
    methods = {
        "Frequency": frequency_scores(sentences),
        "TF-IDF": tfidf_scores(sentences),
        "TextRank": textrank_scores(sentences),
    }

    # 3-5. Summarize, compare, evaluate
    method_stats = {}
    for name, scores in methods.items():
        summary = summarize(sentences, scores, args.top_n)
        stats = compare(text, summary, kw)

        print(f"\n{'=' * 62}\n{name} summary "
              f"({len(summary)} sentences)\n{'=' * 62}")
        print(" ".join(summary))

        r1 = rouge_n(summary, reference, 1) if reference else {"f1": 0.0}
        r2 = rouge_n(summary, reference, 2) if reference else {"f1": 0.0}
        print(f"\n  Compression : {stats['compression_ratio']:.1%} "
              f"({stats['summary_words']} of {stats['original_words']} words)")
        print(f"  Keywords    : kept {stats['keywords_kept']} "
              f"of the article's top 10")
        if reference:
            print(f"  ROUGE-1     : precision {r1['precision']:.2%} | "
                  f"recall {r1['recall']:.2%} | F1 {r1['f1']:.2%}")
            print(f"  ROUGE-2     : precision {r2['precision']:.2%} | "
                  f"recall {r2['recall']:.2%} | F1 {r2['f1']:.2%}")

        method_stats[name] = {
            "rouge1_f1": r1["f1"],
            "compression": stats["compression_ratio"] * 100,
            "summary_sentences": len(summary),
        }

    plot_comparison(method_stats, os.path.join(OUTPUT_DIR,
                                               "method_comparison.png"))
    print(f"\n[EXPORT] Plot -> {OUTPUT_DIR}/method_comparison.png")


if __name__ == "__main__":
    main()