from collections import defaultdict
from typing import Dict, List, Tuple
import re

from langchain_core.documents import Document
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
SECTION_MATCH_BASE_BOOST = 4
SECTION_MATCH_REPEAT_BOOST = 3
SECTION_COVERAGE_BOOST = 2
SECTION_HEADING_MATCH_BOOST = 4
SECTION_GROUP_FALLBACK_PREFIX_LEN = 180
GENERIC_SECTION_BASE_BOOST = 1.5
GENERIC_SECTION_REPEAT_BOOST = 1.25
GENERIC_SECTION_COVERAGE_BOOST = 1.0
GENERIC_SECTION_NEIGHBOR_PAGE_BOOST = 1.5
GENERIC_SECTION_NEIGHBOR_RADIUS = 1
MAX_STRONG_SECTION_COHESION_BOOST = 18
MAX_GENERIC_SECTION_COHESION_BOOST = 8
FACT_FOCUS_BASE_BOOST = 4
LEGAL_PROCESS_BASE_PENALTY = 3
LEGAL_PROCESS_QUERY_BOOST = 5
FACT_QUERY_BOOST = 2
NARRATIVE_FIRST_QUERY_MARKERS = {
    "cerita", "narasi", "kronologi", "timeline", "kejadian", "peristiwa",
    "aktor", "hubungan", "jaringan", "social", "diagram", "investigasi",
}
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


def compact_alpha(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


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


def bm25_like_score(text: str, keywords: List[str]) -> float:
    words = canonicalize_tokens(tokenize(text))
    if not words or not keywords:
        return 0.0

    counts = defaultdict(int)
    for word in words:
        counts[word] += 1

    avg_len = 120.0
    k1 = 1.4
    b = 0.72
    score = 0.0
    doc_len = len(words)
    for keyword in set(keywords):
        tf = counts.get(keyword, 0)
        if tf <= 0:
            continue
        score += ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_len))))
    return score


def narrative_metadata_boost(doc: Document, question_tokens: List[str]) -> float:
    metadata = doc.metadata or {}
    classification = str(metadata.get("classification") or metadata.get("content_type") or "").lower()
    token_set = set(question_tokens)
    asks_narrative = bool(token_set & NARRATIVE_FIRST_QUERY_MARKERS)
    score = 0.0

    if asks_narrative and classification == "narrative":
        score += 8.0
    elif asks_narrative and classification in {"metadata", "legal_reference", "noise"}:
        score -= 5.0
    elif classification == "noise":
        score -= 8.0

    if metadata.get("is_narrative") is True and asks_narrative:
        score += 2.0

    confidence = metadata.get("classification_confidence")
    if isinstance(confidence, (int, float)) and classification == "narrative" and asks_narrative:
        score += float(confidence)

    return score


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


def match_keyword_variants(text: str, compact_text: str, variants: List[str]) -> int:
    score = 0
    for variant in variants:
        normalized_variant = normalize(variant)
        compact_variant = compact_alpha(variant)
        if normalized_variant and normalized_variant in text:
            score += 1
            continue
        if compact_variant and compact_variant in compact_text:
            score += 1
    return score


def classify_document_focus(doc: Document) -> Tuple[int, int]:
    metadata = doc.metadata or {}
    title = metadata.get("section", "")
    content = doc.page_content or ""
    text = normalize(f"{title} {content}")
    compact_text = compact_alpha(f"{title} {content}")

    fact_variants = [
        "bahwa",
        "pada tanggal",
        "kemudian",
        "selanjutnya",
        "saksi",
        "korban",
        "terdakwa",
        "bertemu",
        "menerima",
        "menyerahkan",
        "mentransfer",
        "transfer",
        "uang",
        "hubungan",
        "perkenalan",
        "kronologi",
        "kejadian",
        "peristiwa",
    ]
    legal_variants = [
        "menimbang",
        "mengadili",
        "amar putusan",
        "memutus perkara",
        "pemohon kasasi",
        "termohon kasasi",
        "jaksa penuntut umum",
        "penuntut umum",
        "majelis hakim",
        "hakim ketua",
        "panitera",
        "pengadilan negeri",
        "pengadilan tinggi",
        "membaca tuntutan pidana",
        "dakwaan",
        "putusan",
        "membebaskan terdakwa",
        "menjatuhkan pidana",
        "membebankan biaya perkara",
        "barang bukti berupa",
        "permohonan kasasi",
        "alasan alasan yang diajukan",
        "judex facti",
    ]

    fact_score = match_keyword_variants(text, compact_text, fact_variants)
    legal_score = match_keyword_variants(text, compact_text, legal_variants)

    title_norm = normalize(title)
    if title_norm.startswith("general (page"):
        legal_score = max(0, legal_score - 1)

    return fact_score, legal_score


def query_prefers_legal_process(keywords: List[str], phrases: List[str], raw_question_tokens: List[str]) -> bool:
    query_text = normalize(" ".join(list(raw_question_tokens) + list(phrases) + list(keywords)))
    compact_query = compact_alpha(query_text)
    legal_variants = [
        "putusan",
        "amar",
        "amar putusan",
        "vonis",
        "hukuman",
        "pidana",
        "dakwaan",
        "tuntutan",
        "kasasi",
        "penahanan",
        "majelis hakim",
        "jaksa",
        "pasal",
        "pengadilan",
        "barang bukti",
    ]
    return match_keyword_variants(query_text, compact_query, legal_variants) >= 1


def query_prefers_case_facts(keywords: List[str], phrases: List[str], raw_question_tokens: List[str]) -> bool:
    query_text = normalize(" ".join(list(raw_question_tokens) + list(phrases) + list(keywords)))
    compact_query = compact_alpha(query_text)
    fact_variants = [
        "kasus",
        "fakta",
        "kronologi",
        "ringkasan",
        "rangkuman",
        "aktor",
        "relasi",
        "jaringan sosial",
        "korban",
        "saksi",
        "terdakwa",
        "siapa",
        "kepada siapa",
        "aliran dana",
        "transfer",
        "peristiwa",
        "timeline",
    ]
    return match_keyword_variants(query_text, compact_query, fact_variants) >= 1


def content_focus_adjustment(
    doc: Document,
    keywords: List[str],
    phrases: List[str],
    raw_question_tokens: List[str],
) -> float:
    fact_score, legal_score = classify_document_focus(doc)
    if fact_score == 0 and legal_score == 0:
        return 0.0

    prefers_legal = query_prefers_legal_process(keywords, phrases, raw_question_tokens)
    prefers_facts = query_prefers_case_facts(keywords, phrases, raw_question_tokens)

    adjustment = 0.0
    if prefers_legal:
        adjustment += min(legal_score, 3) * LEGAL_PROCESS_QUERY_BOOST
        adjustment += min(fact_score, 2) * 0.75
        return adjustment

    adjustment += min(fact_score, 3) * FACT_FOCUS_BASE_BOOST

    if prefers_facts:
        adjustment += min(max(fact_score - legal_score, 0), 2) * FACT_QUERY_BOOST

    if legal_score > fact_score:
        adjustment -= min(legal_score - fact_score, 3) * LEGAL_PROCESS_BASE_PENALTY
    elif legal_score > 0 and fact_score == 0:
        adjustment -= min(legal_score, 2) * 1.5

    return adjustment


def section_group_key(doc: Document) -> Tuple[str, str]:
    metadata = doc.metadata or {}
    source = str(metadata.get("source") or metadata.get("workspace_id") or "")
    section = normalize(str(metadata.get("section", "")))

    if section:
        return source, f"section:{section}"

    page = metadata.get("page")
    semantic_prefix = normalize(doc.page_content[:SECTION_GROUP_FALLBACK_PREFIX_LEN])
    if page is not None:
        return source, f"page:{page}:{semantic_prefix}"

    return source, f"content:{semantic_prefix}"


def is_generic_section_title(title: str) -> bool:
    title_norm = normalize(title)
    if not title_norm:
        return True

    generic_patterns = (
        r"^general\s*\(page\s*\d+\)$",
        r"^page\s*\d+$",
        r"^halaman\s*\d+$",
        r"^section\s*\d+$",
    )

    return any(re.match(pattern, title_norm) for pattern in generic_patterns)


def build_section_group_signals(
    docs: List[Document],
    keywords: List[str],
    phrases: List[str],
) -> Dict[Tuple[str, str], Dict[str, object]]:
    groups: Dict[Tuple[str, str], Dict[str, object]] = defaultdict(
        lambda: {
            "doc_count": 0,
            "semantic_indexes": set(),
            "heading_matches": 0,
            "phrase_matches": 0,
        }
    )

    for doc in docs:
        key = section_group_key(doc)
        group = groups[key]
        group["doc_count"] += 1

        semantic_index = doc.metadata.get("semantic_chunk_index")
        if semantic_index is not None:
            group["semantic_indexes"].add(semantic_index)

        title = doc.metadata.get("section", "")
        if title and keywords:
            title_tokens = set(canonicalize_tokens(tokenize(title)))
            keyword_overlap = len(title_tokens & set(keywords))
            if keyword_overlap > 0:
                group["heading_matches"] += 1

        if title and (
            contains_exact_action_phrase(title, keywords)
            or contains_query_phrase(title, phrases)
        ):
            group["phrase_matches"] += 1

    return groups


def build_page_group_signals(
    docs: List[Document],
    keywords: List[str],
    phrases: List[str],
) -> Dict[Tuple[str, int], Dict[str, object]]:
    groups: Dict[Tuple[str, int], Dict[str, object]] = defaultdict(
        lambda: {
            "doc_count": 0,
            "semantic_indexes": set(),
            "heading_matches": 0,
            "phrase_matches": 0,
        }
    )

    for doc in docs:
        metadata = doc.metadata or {}
        page = metadata.get("page")
        if page is None:
            continue

        source = str(metadata.get("source") or metadata.get("workspace_id") or "")
        key = (source, int(page))
        group = groups[key]
        group["doc_count"] += 1

        semantic_index = metadata.get("semantic_chunk_index")
        if semantic_index is not None:
            group["semantic_indexes"].add(semantic_index)

        title = metadata.get("section", "")
        if title and keywords:
            title_tokens = set(canonicalize_tokens(tokenize(title)))
            if title_tokens & set(keywords):
                group["heading_matches"] += 1

        if title and (
            contains_exact_action_phrase(title, keywords)
            or contains_query_phrase(title, phrases)
        ):
            group["phrase_matches"] += 1

    return groups


def section_group_boost(
    doc: Document,
    group_signals: Dict[Tuple[str, str], Dict[str, object]],
    page_signals: Dict[Tuple[str, int], Dict[str, object]],
) -> float:
    metadata = doc.metadata or {}
    title = metadata.get("section", "")
    source = str(metadata.get("source") or metadata.get("workspace_id") or "")
    page = metadata.get("page")

    if is_generic_section_title(title) and page is not None:
        exact_page_group = page_signals.get((source, int(page)))
        if not exact_page_group:
            return 0.0

        doc_count = int(exact_page_group.get("doc_count", 0) or 0)
        semantic_indexes = exact_page_group.get("semantic_indexes", set()) or set()
        phrase_matches = int(exact_page_group.get("phrase_matches", 0) or 0)

        neighbor_pages = set()
        neighbor_doc_count = 0
        for offset in range(-GENERIC_SECTION_NEIGHBOR_RADIUS, GENERIC_SECTION_NEIGHBOR_RADIUS + 1):
            neighbor_page = int(page) + offset
            if neighbor_page < 0:
                continue
            neighbor_group = page_signals.get((source, neighbor_page))
            if not neighbor_group:
                continue
            neighbor_pages.add(neighbor_page)
            neighbor_doc_count += int(neighbor_group.get("doc_count", 0) or 0)

        if doc_count <= 1 and len(semantic_indexes) <= 1 and len(neighbor_pages) <= 1:
            return 0.0

        boost = 0.0
        boost += GENERIC_SECTION_BASE_BOOST
        boost += max(0, doc_count - 1) * GENERIC_SECTION_REPEAT_BOOST
        boost += max(0, len(semantic_indexes) - 1) * GENERIC_SECTION_COVERAGE_BOOST

        if len(neighbor_pages) >= 2:
            boost += GENERIC_SECTION_NEIGHBOR_PAGE_BOOST
            boost += min(len(neighbor_pages) - 2, 1) * 0.75

        if neighbor_doc_count >= 4:
            boost += 1.0

        if phrase_matches >= 1:
            boost += 1.0

        return min(boost, MAX_GENERIC_SECTION_COHESION_BOOST)

    group = group_signals.get(section_group_key(doc))
    if not group:
        return 0.0

    doc_count = int(group.get("doc_count", 0) or 0)
    semantic_indexes = group.get("semantic_indexes", set()) or set()
    heading_matches = int(group.get("heading_matches", 0) or 0)
    phrase_matches = int(group.get("phrase_matches", 0) or 0)

    if doc_count <= 1 and len(semantic_indexes) <= 1:
        return 0.0

    boost = 0.0
    boost += SECTION_MATCH_BASE_BOOST
    boost += max(0, doc_count - 1) * SECTION_MATCH_REPEAT_BOOST
    boost += max(0, len(semantic_indexes) - 1) * SECTION_COVERAGE_BOOST

    if heading_matches >= 2:
        boost += SECTION_HEADING_MATCH_BOOST
    if phrase_matches >= 1:
        boost += 2

    return min(boost, MAX_STRONG_SECTION_COHESION_BOOST)


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

    group_signals = build_section_group_signals(docs, expanded_keywords, query.phrases)
    page_signals = build_page_group_signals(docs, expanded_keywords, query.phrases)

    # 2️⃣ rerank using title + keyword density + section-level cohesion
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
        bm25_score = bm25_like_score(content, expanded_keywords) * 2.5
        narrative_boost = narrative_metadata_boost(doc, raw_question_tokens)
        focus_adjustment = content_focus_adjustment(
            doc,
            expanded_keywords,
            query.phrases,
            raw_question_tokens,
        )

        cohesion_boost = section_group_boost(doc, group_signals, page_signals)
        final_score = (
            base_score
            + title_signal
            + density
            + bm25_score
            + narrative_boost
            + cohesion_boost
            + focus_adjustment
            - penalty
        )

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
