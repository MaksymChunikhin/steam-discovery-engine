import re
from collections import Counter

import pandas as pd

# Minimal English stopword list for bigram cleaning
# Минимальный список англ. стоп-слов для чистки биграмм
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "our", "their", "do", "does", "did", "have", "has", "had", "not", "no",
    "so", "too", "very", "can", "will", "would", "should", "could", "just",
    "than", "then", "there", "here", "out", "up", "down", "about", "all", "any",
    "more", "most", "some", "such", "only", "own", "what", "which", "who",
    "when", "where", "why", "how", "s", "t", "m", "re", "ve", "ll", "d",
}

_WORD_RE = re.compile(r"[a-z']+")


def review_length(reviews: pd.Series, unit: str = "tokens") -> pd.Series:
    # Length of each review in characters or whitespace tokens
    # Длина каждого отзыва в символах или токенах (словах)
    s = reviews.fillna("").astype(str)
    if unit == "chars":
        return s.str.len()
    return s.str.split().apply(len)


def extract_top_bigrams(reviews: pd.Series, n: int = 20):
    # Most frequent bigrams after lowercasing and removing stopwords
    # Самые частые биграммы после lowercase и удаления стоп-слов
    counter = Counter()
    for text in reviews.dropna().astype(str):
        words = [w for w in _WORD_RE.findall(text.lower())
                 if w not in _STOPWORDS and len(w) > 1]
        counter.update(zip(words, words[1:]))
    return counter.most_common(n)
