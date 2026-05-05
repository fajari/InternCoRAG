from typing import List
import re

from langchain.schema import Document
from rag.query import QUERY_STOPWORDS, build_query_analysis, canonicalize_token, tokenize_question
from rag.vectorstore import get_vectorstore


# =====================================================
# CONFIG
# =====================================================

TOP_K = 8
FETCH_K = 20
TITLE_BOOST_WEIGHT = 4
TOC_BOOST_WEIGHT = 3
KEYWORD_DENSITY_WEIGHT = 2
HEADING_BOOST_WEIGHT = 5
HEADING_PHRASE_BOOST = 10
TITLE_GATE_MIN_SCORE = 6
TOC_PRIMARY_WEIGHT = 14
TOC_FULL_MATCH_BONUS = 40
TOC_NEAR_MATCH_BONUS = 24
TOC_MIN_PRIORITY_SCORE = 16
SYNONYM_MAP = {
    "add": {"add", "create", "insert", "register", "new"},
    "create": {"create", "add", "insert", "register", "new"},
    "insert": {"insert", "add", "create"},
    "new": {"new", "add", "create"},
    "edit": {"edit", "update", "modify", "change", "revise"},
    "update": {"update", "edit", "modify", "change", "revise"},
    "modify": {"modify", "edit", "update", "change"},
    "change": {"change", "edit", "update", "modify"},
    "delete": {"delete", "remove", "erase", "drop"},
    "remove": {"remove", "delete", "erase", "drop"},
    "erase": {"erase", "delete", "remove"},
    "record": {"record", "records", "rr"},
    "records": {"records", "record", "rr"},
    "rr": {"rr", "record", "records", "resource"},
    "resource": {"resource", "resources", "rr"},
    "resources": {"resources", "resource", "rr"},
    "flush": {"flush", "clear", "purge", "refresh"},
    "clear": {"clear", "flush", "purge"},
    "resolve": {"resolve", "test", "lookup", "check", "verify"},
    "test": {"test", "verify", "resolve", "check"},
    "verify": {"verify", "test", "check", "resolve"},
}


# =====================================================
# UTIL
# =====================================================

def tokenize(text: str) -> List[str]:
    return tokenize_question(text)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def extract_internal_headings(text: str) -> List[str]:
    if not text:
        return []

    pattern = re.compile(
        r"(?m)^\s*((?:\d+\.)+\d+)\s+([A-Z][A-Za-z0-9/&()'., -]{2,140}?)\s*$"
    )
    headings: List[str] = []
    seen = set()

    for match in pattern.finditer(text):
        heading = f"{match.group(1)} {match.group(2)}".strip()
        heading_norm = normalize(heading)
        if heading_norm and heading_norm not in seen:
            seen.add(heading_norm)
            headings.append(heading)

    return headings


def extract_keywords(question: str) -> List[str]:
    return build_query_analysis(question).terms


def canonicalize_tokens(tokens: List[str]) -> List[str]:
    return [canonicalize_token(token) for token in tokens if token]


def expand_keywords(keywords: List[str]) -> List[str]:
    expanded: List[str] = []
    seen = set()

    for keyword in keywords:
        candidates = [keyword, *sorted(SYNONYM_MAP.get(keyword, set()))]
        for candidate in candidates:
            candidate = canonicalize_token(candidate)
            if candidate in QUERY_STOPWORDS or candidate in seen:
                continue
            seen.add(candidate)
            expanded.append(candidate)

    return expanded


def section_depth(section_title: str) -> int:
    match = re.match(r"^\s*((?:\d+\.)*\d+)", section_title or "")
    if not match:
        return 0
    return match.group(1).count(".")


def extract_primary_action(tokens: List[str]) -> str:
    primary_actions = ("add", "edit", "delete", "create", "update", "remove")
    token_set = set(tokens)

    for action in primary_actions:
        if action in token_set:
            return action

    return ""


def keyword_density_score(text: str, keywords: List[str]) -> float:
    words = canonicalize_tokens(tokenize(text))
    if not words:
        return 0.0

    count = sum(words.count(k) for k in keywords)
    return count / len(words)


def title_boost_score(title: str, keywords: List[str]) -> int:
    title_tokens = set(canonicalize_tokens(tokenize(title)))
    overlap = len(title_tokens & set(keywords))
    score = overlap * TITLE_BOOST_WEIGHT

    if overlap == len(set(keywords)) and overlap > 0:
        score += 6
        score += section_depth(title) * 2

    return score


def heading_boost_score(headings: List[str], keywords: List[str]) -> int:
    if not headings or not keywords:
        return 0

    keyword_set = set(keywords)
    best_score = 0

    for heading in headings:
        heading_tokens = set(canonicalize_tokens(tokenize(heading)))
        overlap = len(heading_tokens & keyword_set)
        score = overlap * HEADING_BOOST_WEIGHT

        if overlap == len(keyword_set) and overlap > 0:
            score += 8
            score += section_depth(heading) * 2

        if contains_exact_action_phrase(heading, keywords):
            score += HEADING_PHRASE_BOOST

        best_score = max(best_score, score)

    return best_score


def toc_boost_score(title: str, toc_entries: List[str], keywords: List[str]) -> int:
    best_score = 0
    title_norm = normalize(title)
    keyword_set = set(keywords)

    for entry in toc_entries:
        entry_norm = normalize(entry)
        entry_tokens = set(canonicalize_tokens(tokenize(entry_norm)))
        overlap = len(keyword_set & entry_tokens)
        score = 0

        if overlap == len(keyword_set) and overlap > 0:
            score += TOC_BOOST_WEIGHT * 3
        elif overlap >= 2:
            score += TOC_BOOST_WEIGHT * 2
        elif overlap == 1:
            score += TOC_BOOST_WEIGHT

        # Strongest TOC signal only when the matching TOC entry
        # actually represents this section, not just one of its children.
        if score > 0 and entry_norm == title_norm:
            score += 6
        elif score > 0 and title_norm.startswith(f"{entry_norm}."):
            score += 2

        best_score = max(best_score, score)

    return best_score


def score_toc_entry(entry: str, keywords: List[str], phrases: List[str]) -> int:
    if not entry:
        return 0

    entry_tokens = set(canonicalize_tokens(tokenize(entry)))
    keyword_set = set(keywords)
    overlap = len(entry_tokens & keyword_set)
    score = overlap * TOC_PRIMARY_WEIGHT

    if keyword_set and keyword_set.issubset(entry_tokens):
        score += TOC_FULL_MATCH_BONUS
    elif keyword_set:
        coverage = overlap / max(len(keyword_set), 1)
        if coverage >= 0.7:
            score += TOC_NEAR_MATCH_BONUS

    if contains_exact_action_phrase(entry, keywords):
        score += 12

    if contains_query_phrase(entry, phrases):
        score += 14

    return score


def best_toc_match(toc_entries: List[str], keywords: List[str], phrases: List[str]) -> tuple[int, str]:
    best_score = 0
    best_entry = ""

    for entry in toc_entries or []:
        score = score_toc_entry(entry, keywords, phrases)
        if score > best_score:
            best_score = score
            best_entry = entry

    return best_score, best_entry


def contains_exact_action_phrase(text: str, keywords: List[str]) -> bool:
    if not keywords:
        return False

    canonical_text_tokens = canonicalize_tokens(tokenize(text))
    canonical_text = " ".join(canonical_text_tokens)

    for size in range(min(3, len(keywords)), 1, -1):
        for start in range(0, len(keywords) - size + 1):
            phrase = " ".join(keywords[start:start + size])
            if phrase and phrase in canonical_text:
                return True

    return False


def contains_query_phrase(text: str, phrases: List[str]) -> bool:
    normalized = normalize(text)
    if not normalized:
        return False
    return any(phrase and phrase in normalized for phrase in phrases)


def intent_penalty_score(title: str, raw_question_tokens: List[str]) -> int:
    title_tokens = set(tokenize(title))
    query_tokens = set(raw_question_tokens)
    penalty = 0

    if "after" in title_tokens and "after" not in query_tokens:
        penalty += 8

    primary_actions = {"add", "edit", "delete", "create", "update", "remove"}
    matched_actions = title_tokens & primary_actions
    query_actions = query_tokens & primary_actions

    if len(matched_actions) >= 2 and len(query_actions) <= 1:
        penalty += 3

    query_action = extract_primary_action(list(query_tokens))
    title_action = extract_primary_action(list(title_tokens))

    if query_action and title_action and query_action != title_action:
        allowed_pairs = {
            ("add", "create"),
            ("create", "add"),
            ("edit", "update"),
            ("update", "edit"),
            ("delete", "remove"),
            ("remove", "delete"),
        }
        if (query_action, title_action) not in allowed_pairs:
            penalty += 12

    return penalty


# =====================================================
# MAIN RETRIEVE FUNCTION
# =====================================================

def retrieve_documents(workspace_id: str, question: str) -> List[Document]:

    vectorstore = get_vectorstore()

    # 1️⃣ similarity search with metadata filter
    docs = vectorstore.similarity_search(
        question,
        k=FETCH_K,
        filter={
            "workspace_id": workspace_id
        }
    )

    if not docs:
        return []

    query = build_query_analysis(question)
    keywords = canonicalize_tokens(query.terms)
    expanded_keywords = expand_keywords(keywords)
    raw_question_tokens = tokenize(question)

    scored_docs = []
    toc_priority_docs = []

    # STRICT TITLE MATCH OVERRIDE
    strict_candidates = []
    for rank, doc in enumerate(docs):
        title = doc.metadata.get("section", "")
        title_tokens = set(canonicalize_tokens(tokenize(title)))
        internal_headings = extract_internal_headings(doc.page_content)

        if keywords and all(k in title_tokens for k in keywords):
            strict_candidates.append((
                contains_exact_action_phrase(title, keywords) or contains_query_phrase(title, query.phrases),
                len(title_tokens & set(keywords)),
                section_depth(title),
                -rank,
                doc,
            ))
            continue

        for heading in internal_headings:
            heading_tokens = set(canonicalize_tokens(tokenize(heading)))
            if keywords and all(k in heading_tokens for k in keywords):
                strict_candidates.append((
                    contains_exact_action_phrase(heading, keywords) or contains_query_phrase(heading, query.phrases),
                    len(heading_tokens & set(keywords)),
                    section_depth(heading),
                    -rank,
                    doc,
                ))
                break

    if strict_candidates:
        strict_candidates.sort(reverse=True, key=lambda item: item[:4])
        return [strict_candidates[0][4]]

    # 2️⃣ rerank using title + keyword density
    for rank, doc in enumerate(docs):

        title = doc.metadata.get("section", "")
        content = doc.page_content
        toc_entries = doc.metadata.get("toc_entries", [])
        internal_headings = extract_internal_headings(content)
        toc_score, matched_toc = best_toc_match(toc_entries, keywords, query.phrases)


        # Base score (inverse rank bias)
        base_score = (TOP_K - rank)

        boost = title_boost_score(title, expanded_keywords)
        toc_boost = toc_boost_score(title, toc_entries, expanded_keywords) + toc_score
        phrase_boost = 12 if (
            contains_exact_action_phrase(title, keywords)
            or contains_query_phrase(title, query.phrases)
        ) else 0
        heading_boost = heading_boost_score(internal_headings, expanded_keywords)
        if query.phrases and any(contains_query_phrase(heading, query.phrases) for heading in internal_headings):
            heading_boost += HEADING_PHRASE_BOOST
        penalty = intent_penalty_score(title, raw_question_tokens)
        title_signal = boost + toc_boost + phrase_boost + heading_boost
        density = 0.0
        if title_signal >= TITLE_GATE_MIN_SCORE:
            density = keyword_density_score(content, expanded_keywords) * KEYWORD_DENSITY_WEIGHT

        final_score = base_score + title_signal + density - penalty

        scored_docs.append((final_score, doc))
        if toc_score >= TOC_MIN_PRIORITY_SCORE:
            toc_priority_docs.append((toc_score, final_score, -rank, matched_toc, doc))

    if toc_priority_docs:
        toc_priority_docs.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        prioritized = [doc for _, _, _, _, doc in toc_priority_docs[:TOP_K]]

        seen_ids = {id(doc) for doc in prioritized}
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        for _, doc in scored_docs:
            if id(doc) in seen_ids:
                continue
            prioritized.append(doc)
            seen_ids.add(id(doc))
            if len(prioritized) >= TOP_K:
                break

        return prioritized

    # 3️⃣ sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # 4️⃣ return only documents
    return [doc for _, doc in scored_docs[:TOP_K]]
