from __future__ import annotations

from dataclasses import dataclass
import re


QUERY_STOPWORDS = {
    "how", "what", "when", "where", "which", "who", "why",
    "tell", "me", "about", "to", "do", "does", "is", "are",
    "the", "a", "an", "of", "for", "on", "in", "after",
}


@dataclass(frozen=True)
class QueryAnalysis:
    original: str
    normalized: str
    terms: list[str]
    phrases: list[str]
    summary: str


def tokenize_question(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", (text or "").lower())


def canonicalize_token(token: str) -> str:
    token = (token or "").lower().strip()
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def build_query_analysis(question: str) -> QueryAnalysis:
    normalized = normalize_question(question)
    raw_tokens = tokenize_question(question)
    canonical_tokens = [canonicalize_token(token) for token in raw_tokens]
    terms = [token for token in canonical_tokens if token and token not in QUERY_STOPWORDS]
    if not terms:
        terms = [token for token in canonical_tokens if token]

    phrases = extract_query_phrases(canonical_tokens)
    summary = summarize_query(terms, phrases)

    return QueryAnalysis(
        original=question or "",
        normalized=normalized,
        terms=terms,
        phrases=phrases,
        summary=summary,
    )


def extract_query_phrases(tokens: list[str]) -> list[str]:
    filtered = [token for token in tokens if token and token not in QUERY_STOPWORDS]
    if not filtered:
        return []

    phrases: list[str] = []
    seen = set()

    for size in (4, 3, 2):
        if len(filtered) < size:
            continue
        for start in range(0, len(filtered) - size + 1):
            window = filtered[start:start + size]
            unique_terms = {token for token in window if len(token) > 1}
            if len(unique_terms) < 2:
                continue
            phrase = " ".join(window).strip()
            if phrase and phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)

    return phrases[:8]


def summarize_query(terms: list[str], phrases: list[str]) -> str:
    if phrases:
        return phrases[0]
    return " ".join(terms[:4]).strip()

