import re
from typing import List
from qdrant_client.http import models
from config import QDRANT_COLLECTION
from rag.query import QueryAnalysis, build_query_analysis, canonicalize_token, tokenize_question
from rag.retriever import retrieve_documents
from rag.helpers import build_sources
from rag.vectorstore import client

NO_ANSWER = "I don't know based on the provided documents."
MAX_KEYPOINTS = 10
SUBSECTION_TITLE_WEIGHT = 10
SUBSECTION_TITLE_PHRASE_WEIGHT = 5
SUBSECTION_BODY_WEIGHT = 2
DOC_TITLE_WEIGHT = 12
DOC_TITLE_PHRASE_WEIGHT = 6
DOC_TEXT_WEIGHT = 4
DOC_RAW_WEIGHT = 1
DOC_FULL_TITLE_MATCH_BONUS = 18
SUBSECTION_FULL_TITLE_MATCH_BONUS = 14
TITLE_DOMINANCE_WEIGHT = 8
TITLE_COVERAGE_FULL_BONUS = 22
TITLE_COVERAGE_NEAR_BONUS = 12
ESTIMATION_TIME_REGEX = re.compile(
    r"(?i)\bestimation time(?:\s+in\s+total)?\s*:\s*.*"
)
SUBSECTION_REGEX = re.compile(
    r"(?<!\w)((?:\d+\.)+\d+)\s+([A-Z][A-Za-z0-9/&()' -]{2,120}?)"
    r"(?=\s+(?:Estimation time in total:|NO ACTION|$))"
)
MULTILINE_SUBSECTION_REGEX = re.compile(
    r"(?m)^\s*((?:\d+\.)+\d+)\s+([A-Z][A-Za-z0-9/&()'., -]{2,140}?)\s*$"
)


# ============================================================
# CLEANING UTILITIES
# ============================================================

def strip_estimation_time(text: str) -> str:
    if not text:
        return ""

    cleaned_lines = []
    for raw_line in text.splitlines():
        if ESTIMATION_TIME_REGEX.search(raw_line.strip()):
            continue
        cleaned_lines.append(raw_line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = ESTIMATION_TIME_REGEX.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()

def clean_text(text: str) -> str:
    """
    Clean numbering artifacts and normalize spacing.
    """

    if not text:
        return ""

    text = strip_estimation_time(text)

    # Remove numbering like:
    # 8.3 Termination
    # 3 Termination
    text = re.sub(r"\n?\s*\d+(\.\d+)*\s+(?=[A-Z])", "\n", text)

    # Remove stray page number at end
    text = re.sub(r"\s\d+\s*$", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def remove_duplicate_sentences(items: List[str]) -> List[str]:
    seen = set()
    result = []

    for item in items:
        key = item.strip().lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())

    return result


def extract_query_terms(question: str) -> List[str]:
    return build_query_analysis(question).terms


def keyword_overlap_score(text: str, query_terms: List[str]) -> int:
    if not text or not query_terms:
        return 0

    tokens = {
        canonicalize_token(token)
        for token in tokenize_question(text)
    }
    return len(tokens & set(query_terms))


def full_query_match_bonus(text: str, query_terms: List[str], bonus: int) -> int:
    if not text or not query_terms:
        return 0

    tokens = {
        canonicalize_token(token)
        for token in tokenize_question(text)
    }
    query_set = set(query_terms)
    if query_set and query_set.issubset(tokens):
        return bonus
    return 0


def title_dominance_bonus(text: str, query_terms: List[str]) -> int:
    if not text or not query_terms:
        return 0

    tokens = {
        canonicalize_token(token)
        for token in tokenize_question(text)
    }
    query_set = set(query_terms)
    if not tokens or not query_set:
        return 0

    overlap = len(tokens & query_set)
    if overlap <= 0:
        return 0

    coverage = overlap / max(len(query_set), 1)
    score = overlap * TITLE_DOMINANCE_WEIGHT

    if coverage >= 1.0:
        score += TITLE_COVERAGE_FULL_BONUS
    elif coverage >= 0.7:
        score += TITLE_COVERAGE_NEAR_BONUS

    return score


def toc_entry_score(entry: str, query: QueryAnalysis) -> int:
    if not entry:
        return 0

    score = keyword_overlap_score(entry, query.terms) * 14
    score += phrase_match_score(entry, query.terms, query.phrases) * 2
    score += full_query_match_bonus(entry, query.terms, 28)
    score += title_dominance_bonus(entry, query.terms)
    return score


def best_toc_entry(entries: List[str], query: QueryAnalysis) -> tuple[int, str]:
    best_score = 0
    best_entry = ""

    for entry in entries or []:
        score = toc_entry_score(entry, query)
        if score > best_score:
            best_score = score
            best_entry = entry

    return best_score, best_entry


def phrase_match_score(text: str, query_terms: List[str], query_phrases: List[str] | None = None) -> int:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not normalized or not query_terms:
        return 0

    score = 0
    for phrase in query_phrases or []:
        if phrase and phrase in normalized:
            score = max(score, len(phrase.split()) * 6)

    max_size = min(4, len(query_terms))
    for size in range(max_size, 1, -1):
        for start in range(0, len(query_terms) - size + 1):
            phrase = " ".join(query_terms[start:start + size]).strip()
            if phrase and phrase in normalized:
                score = max(score, size * 4)

    return score


def select_relevant_subsection(
    text: str,
    query: QueryAnalysis,
    default_title: str,
) -> tuple[str, str]:
    matches = list(SUBSECTION_REGEX.finditer(text or ""))
    if not matches:
        return default_title, text

    query_terms = query.terms
    candidates: List[tuple[int, int, str, str]] = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        section_title = f"{match.group(1)} {match.group(2)}".strip()

        title_score = keyword_overlap_score(section_title, query_terms) * SUBSECTION_TITLE_WEIGHT
        title_phrase_score = phrase_match_score(section_title, query_terms, query.phrases) * SUBSECTION_TITLE_PHRASE_WEIGHT
        title_full_match_bonus = full_query_match_bonus(
            section_title,
            query_terms,
            SUBSECTION_FULL_TITLE_MATCH_BONUS,
        )
        title_dominance = title_dominance_bonus(section_title, query_terms)
        body_score = keyword_overlap_score(section_text[:1500], query_terms) * SUBSECTION_BODY_WEIGHT
        score = title_score + title_phrase_score + title_full_match_bonus + title_dominance + body_score

        if score <= 0:
            continue

        candidates.append((score, len(section_text), section_title, section_text))

    if not candidates:
        return default_title, text

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, title, section_text = candidates[0]
    return title, section_text


def select_best_doc_match(docs: List, question: str):
    if not docs:
        return None, "", ""

    query = build_query_analysis(question)
    query_terms = query.terms
    scored_candidates: List[tuple[int, int, object, str, str]] = []

    for index, doc in enumerate(docs):
        title = doc.metadata.get("section", "Policy Section")
        raw_text = doc.page_content or ""
        selected_title, selected_text = select_relevant_subsection(raw_text, query, title)
        toc_score, matched_toc_entry = best_toc_entry(doc.metadata.get("toc_entries", []), query)

        title_score = keyword_overlap_score(selected_title, query_terms) * DOC_TITLE_WEIGHT
        text_score = keyword_overlap_score(selected_text[:2000], query_terms) * DOC_TEXT_WEIGHT
        raw_score = keyword_overlap_score(raw_text[:2000], query_terms) * DOC_RAW_WEIGHT
        phrase_score = phrase_match_score(selected_title, query_terms, query.phrases) * DOC_TITLE_PHRASE_WEIGHT
        phrase_score += phrase_match_score(selected_text[:2000], query_terms, query.phrases)
        phrase_score += phrase_match_score(matched_toc_entry, query_terms, query.phrases)
        full_title_bonus = full_query_match_bonus(
            selected_title,
            query_terms,
            DOC_FULL_TITLE_MATCH_BONUS,
        )
        title_dominance = title_dominance_bonus(selected_title, query_terms)

        score = (
            title_score
            + text_score
            + raw_score
            + phrase_score
            + full_title_bonus
            + title_dominance
            + toc_score
        )
        scored_candidates.append((score, -index, doc, selected_title, selected_text))

    scored_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, best_doc, best_title, best_text = scored_candidates[0]
    return best_doc, best_title, best_text


def is_general_section_title(title: str) -> bool:
    normalized = (title or "").strip().lower()
    return normalized.startswith("general (page ")


def fetch_source_documents(workspace_id: str, source: str) -> List[dict]:
    if not workspace_id or not source:
        return []

    points, _ = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.workspace_id",
                    match=models.MatchValue(value=str(workspace_id)),
                ),
                models.FieldCondition(
                    key="metadata.source",
                    match=models.MatchValue(value=str(source)),
                ),
            ]
        ),
        with_payload=True,
        limit=512,
    )

    docs: List[dict] = []
    for point in points:
        payload = point.payload or {}
        metadata = payload.get("metadata", {}) or {}
        page_content = payload.get("page_content", "") or ""
        docs.append({
            "page_content": page_content,
            "metadata": metadata,
        })

    docs.sort(
        key=lambda item: (
            item["metadata"].get("page", 0),
            0 if not is_general_section_title(item["metadata"].get("section", "")) else 1,
        )
    )
    return docs


def collect_subsection_across_pages(
    workspace_id: str,
    best_doc,
    selected_title: str,
    selected_text: str,
) -> tuple[str, List[dict]]:
    source = best_doc.metadata.get("source")
    start_page = best_doc.metadata.get("page", 0)
    source_docs = fetch_source_documents(workspace_id, source)
    if not source_docs:
        return best_doc.page_content, [best_doc]

    aggregated_texts = [selected_text or best_doc.page_content]
    source_refs = [best_doc]
    started = False
    last_page = start_page

    for item in source_docs:
        metadata = item.get("metadata", {})
        page = metadata.get("page", 0)
        section = metadata.get("section", "")
        content = item.get("page_content", "")

        if page < start_page:
            continue

        if not started:
            same_page = page == start_page
            same_section = normalize_heading_text(section) == normalize_heading_text(selected_title)
            same_content = content.strip() == (best_doc.page_content or "").strip()
            if same_page and (same_section or same_content):
                started = True
            continue

        if page == start_page:
            continue

        if page > last_page + 1:
            break

        if section and not is_general_section_title(section):
            if normalize_heading_text(section) != normalize_heading_text(selected_title):
                break

        if content.strip():
            aggregated_texts.append(content)
            source_refs.append(item)
            last_page = page

    combined = "\n".join(text for text in aggregated_texts if text.strip()).strip()
    return combined or best_doc.page_content, source_refs


def select_highlight_snippets(
    question: str,
    section_title: str,
    source_text: str,
    keypoints: List[str],
) -> List[str]:
    query_terms = extract_query_terms(question)
    query = build_query_analysis(question)
    candidates: List[tuple[float, str]] = []
    normalized_keypoints = [
        re.sub(r"\s+", " ", keypoint).strip(" .:-")
        for keypoint in keypoints
        if keypoint and keypoint.strip()
    ]
    source_sentences = split_into_sentences(source_text)

    for sentence in source_sentences:
        normalized = re.sub(r"\s+", " ", sentence).strip(" .:-")
        if not is_useful_keypoint(normalized):
            continue

        score = keyword_overlap_score(normalized, query_terms) * 4
        score += phrase_match_score(normalized, query_terms, query.phrases) * 2
        score += min(len(normalized.split()), 28) / 10

        if section_title:
            normalized_title = normalize_heading_text(section_title)
            if normalized_title and normalized_title in normalize_heading_text(normalized):
                score += 3

        for keypoint in normalized_keypoints:
            normalized_keypoint = normalize_heading_text(keypoint)
            normalized_sentence = normalize_heading_text(normalized)
            if not normalized_keypoint or not normalized_sentence:
                continue
            if normalized_keypoint in normalized_sentence:
                score += 10
            elif normalized_sentence in normalized_keypoint:
                score += 5
            else:
                overlap = keyword_overlap_score(normalized, extract_query_terms(keypoint))
                score += overlap * 1.5

        candidates.append((score, normalized))

    normalized_lines = [
        re.sub(r"\s+", " ", line).strip(" .:-")
        for line in source_text.splitlines()
    ]
    for line in normalized_lines:
        if len(line) < 18 or not is_useful_keypoint(line):
            continue
        score = keyword_overlap_score(line, query_terms) * 3
        score += phrase_match_score(line, query_terms, query.phrases)
        score += min(len(line.split()), 24) / 12
        candidates.append((score, line))

    if section_title and is_useful_keypoint(section_title):
        candidates.append((keyword_overlap_score(section_title, query_terms) + 2, section_title))

    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    selected: List[str] = []
    seen = set()
    for _, text in candidates:
        normalized = text.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        selected.append(text)
        if len(selected) >= 3:
            break

    if selected:
        return selected[:2]

    fallback = [
        re.sub(r"\s+", " ", keypoint).strip(" .:-")
        for keypoint in keypoints
        if is_useful_keypoint(keypoint)
    ]
    return fallback[:2]


def build_keyword_summary(
    question: str,
    section_title: str,
    source_text: str,
    keypoints: List[str],
) -> str:
    query_terms = extract_query_terms(question)
    query = build_query_analysis(question)
    sentence_candidates = split_into_sentences(source_text)
    scored_sentences: List[tuple[float, str]] = []

    for sentence in sentence_candidates:
        normalized = re.sub(r"\s+", " ", sentence).strip(" .:-")
        if not is_useful_keypoint(normalized):
            continue

        score = keyword_overlap_score(normalized, query_terms) * 4
        score += phrase_match_score(normalized, query_terms, query.phrases) * 2
        score += min(len(normalized.split()), 24) / 12
        scored_sentences.append((score, normalized))

    scored_sentences.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    selected: List[str] = []
    seen = set()
    for _, sentence in scored_sentences:
        key = sentence.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) >= 2:
            break

    if not selected:
        selected = keypoints[:2]

    summary_parts = []
    if section_title:
        summary_parts.append(section_title)
    if selected:
        summary_parts.append(" ".join(summarize_keypoint(item, max_words=20) for item in selected))

    summary = " ".join(part.strip() for part in summary_parts if part.strip()).strip()
    return summary if summary else build_overview(keypoints)


def extract_nested_subsections(
    text: str,
    parent_title: str | None = None,
) -> List[tuple[str, str]]:
    matches = list(MULTILINE_SUBSECTION_REGEX.finditer(text or ""))
    if not matches:
        return []

    parent_normalized = normalize_heading_text(parent_title)
    subsections: List[tuple[str, str]] = []

    for index, match in enumerate(matches):
        subsection_title = f"{match.group(1)} {match.group(2)}".strip()
        subsection_normalized = normalize_heading_text(subsection_title)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        subsection_body = (text[start:end] or "").strip()

        if subsection_normalized == parent_normalized:
            continue

        if not subsection_body:
            continue

        subsections.append((subsection_title, subsection_body))

    return subsections


def build_subsection_summaries(
    question: str,
    title: str,
    raw_text: str,
) -> List[dict]:
    subsection_blocks = extract_nested_subsections(raw_text, title)
    if not subsection_blocks:
        subsection_blocks = [(title, raw_text)]

    query_terms = extract_query_terms(question)
    query = build_query_analysis(question)
    subsection_candidates: List[tuple[float, int, dict]] = []

    for subsection_title, subsection_text in subsection_blocks:
        subsection_keypoints = extract_keypoints(subsection_text, subsection_title, question)
        subsection_summary = build_keyword_summary(
            question,
            subsection_title,
            subsection_text,
            subsection_keypoints,
        )

        if not subsection_summary and not subsection_keypoints:
            continue

        subsection_payload = {
            "title": subsection_title,
            "text": subsection_text,
            "summary": subsection_summary,
            "keypoints": subsection_keypoints,
        }
        score = keyword_overlap_score(subsection_title, query_terms) * SUBSECTION_TITLE_WEIGHT
        score += phrase_match_score(subsection_title, query_terms, query.phrases) * SUBSECTION_TITLE_PHRASE_WEIGHT
        score += full_query_match_bonus(
            subsection_title,
            query_terms,
            SUBSECTION_FULL_TITLE_MATCH_BONUS,
        )
        score += title_dominance_bonus(subsection_title, query_terms)
        score += keyword_overlap_score(subsection_text[:1500], query_terms) * SUBSECTION_BODY_WEIGHT
        score += phrase_match_score(subsection_text[:1500], query_terms, query.phrases)

        if subsection_summary:
            score += keyword_overlap_score(subsection_summary, query_terms) * 2
            score += phrase_match_score(subsection_summary, query_terms, query.phrases)

        if subsection_keypoints:
            joined_keypoints = " ".join(subsection_keypoints[:3])
            score += keyword_overlap_score(joined_keypoints, query_terms) * 2
            score += phrase_match_score(joined_keypoints, query_terms, query.phrases)

        subsection_candidates.append((score, len(subsection_text), subsection_payload))

    if not subsection_candidates:
        return []

    positive_candidates = [item for item in subsection_candidates if item[0] > 0]
    ranked_candidates = positive_candidates if positive_candidates else subsection_candidates
    ranked_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    return [ranked_candidates[0][2]]


def build_resume_points(
    question: str,
    source_text: str,
    keypoints: List[str],
) -> List[str]:
    query_terms = extract_query_terms(question)
    query = build_query_analysis(question)
    candidates = keypoints[:] if keypoints else split_into_sentences(source_text)
    scored: List[tuple[float, str]] = []

    for candidate in candidates:
        normalized = re.sub(r"\s+", " ", candidate).strip(" .:-")
        if not is_useful_keypoint(normalized):
            continue
        score = sentence_information_score(normalized, query_terms)
        score += phrase_match_score(normalized, query_terms, query.phrases) * 1.5
        scored.append((score, normalized))

    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    selected: List[str] = []
    seen = set()
    for _, item in scored:
        cleaned = summarize_keypoint(item, max_words=30)
        if not is_useful_keypoint(cleaned):
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        selected.append(cleaned)
        if len(selected) >= 4:
            break

    return selected


def build_resume_points_from_subsections(
    question: str,
    subsection_summaries: List[dict],
    fallback_text: str,
    fallback_keypoints: List[str],
) -> List[str]:
    subsection_candidates: List[str] = []

    for subsection in subsection_summaries:
        summary = (subsection.get("summary") or "").strip()
        subsection_keypoints = subsection.get("keypoints") or []

        if summary:
            subsection_candidates.append(summary)

        subsection_candidates.extend(subsection_keypoints[:3])

    if subsection_candidates:
        return build_resume_points(question, "\n".join(subsection_candidates), subsection_candidates)

    return build_resume_points(question, fallback_text, fallback_keypoints)


def rank_highlight_pages(
    source_refs: List,
    question: str,
    highlights: List[str],
    selected_title: str = "",
) -> List[int]:
    query_terms = extract_query_terms(question)
    query = build_query_analysis(question)
    scored_pages: List[tuple[int, int]] = []
    seen_pages = set()
    normalized_selected_title = normalize_heading_text(selected_title)

    for fallback_index, item in enumerate(source_refs):
        metadata = item.get("metadata", {}) if isinstance(item, dict) else item.metadata
        page = metadata.get("page")
        if not isinstance(page, int) or page in seen_pages:
            continue

        seen_pages.add(page)
        content = item.get("page_content", "") if isinstance(item, dict) else item.page_content
        section = metadata.get("section", "")
        score = keyword_overlap_score(content, query_terms) * 3
        score += phrase_match_score(content, query_terms, query.phrases) * 2

        normalized_section = normalize_heading_text(section)
        if normalized_selected_title and normalized_section == normalized_selected_title:
            score += 20
            score += title_dominance_bonus(section, query_terms)
        elif normalized_selected_title and normalized_selected_title in normalize_heading_text(content[:800]):
            score += 8

        normalized_content = re.sub(r"\s+", " ", (content or "").lower()).strip()
        for highlight in highlights:
            normalized_highlight = re.sub(r"\s+", " ", (highlight or "").lower()).strip()
            if not normalized_highlight:
                continue
            if normalized_highlight in normalized_content:
                score += 18
            score += keyword_overlap_score(highlight, query_terms)

        scored_pages.append((score, -fallback_index, page))

    scored_pages.sort(key=lambda item: (item[0], item[1]), reverse=True)
    ordered_pages = [page for _, _, page in scored_pages]

    if ordered_pages:
        return ordered_pages

    return [
        metadata.get("page")
        for item in source_refs
        for metadata in [item.get("metadata", {}) if isinstance(item, dict) else item.metadata]
        if isinstance(metadata.get("page"), int)
    ]


def split_long_statement(text: str) -> List[str]:
    """
    Split a long policy paragraph into readable bullet points.
    """

    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text).strip(" .:-")

    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+|;\s+", normalized)
    cleaned_parts = []

    for part in parts:
        part = part.strip(" .:-")
        if len(part) >= 12:
            cleaned_parts.append(part)

    return cleaned_parts


def split_into_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip(" .:-")
    if not normalized:
        return []

    parts = re.split(r"(?<=[.!?])\s+|;\s+|(?<=:)\s+(?=[A-Z])", normalized)
    return [part.strip(" .:-") for part in parts if part.strip(" .:-")]


def sentence_information_score(text: str, query_terms: List[str] | None = None) -> float:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return -1.0

    if ESTIMATION_TIME_REGEX.search(normalized):
        return -1.0

    words = tokenize_question(normalized)
    lowered = normalized.lower()
    score = min(len(words), 24) / 4.0

    action_terms = {
        "must", "should", "required", "require", "ensure", "verify",
        "check", "update", "create", "delete", "remove", "submit",
        "record", "complete", "review", "confirm", "notify", "enter",
        "select", "attach", "provide",
    }
    score += sum(1.2 for word in words if word.lower() in action_terms)

    if ":" in normalized:
        score += 1.0

    if any(token.isdigit() for token in words):
        score += 0.4

    if query_terms:
        score += keyword_overlap_score(normalized, query_terms) * 2.5

    weak_starts = {
        "note", "example", "remarks", "remark", "time", "no", "status",
        "description", "details",
    }
    first_word = words[0].lower() if words else ""
    if first_word in weak_starts:
        score -= 1.5

    if len(words) < 5:
        score -= 2.0

    if len(words) > 32:
        score -= 1.0

    if lowered.startswith(("if ", "when ", "after ", "before ")):
        score += 0.8

    return score


def rank_candidate_points(points: List[str], query_terms: List[str] | None = None) -> List[str]:
    scored: List[tuple[float, str]] = []
    for point in remove_duplicate_sentences(points):
        if not is_useful_keypoint(point):
            continue
        summary = summarize_keypoint(point, max_words=24)
        if not is_useful_keypoint(summary):
            continue
        scored.append((sentence_information_score(summary, query_terms), summary))

    scored.sort(key=lambda item: (item[0], len(item[1])), reverse=True)

    ordered: List[str] = []
    seen = set()
    for _, point in scored:
        normalized = point.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(point)

    return ordered


def summarize_keypoint(text: str, max_words: int = 28) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip(" .:-|")
    if not normalized:
        return ""

    normalized = re.sub(r"^\d+(\.\d+)*\s*", "", normalized)
    sentence = split_into_sentences(normalized)[0] if split_into_sentences(normalized) else ""
    sentence = sentence.strip(" .")
    if not sentence:
        return ""

    if ":" in sentence:
        left, right = sentence.split(":", 1)
        left = left.strip()
        right_words = right.strip().split()
        if left and right_words and len(left.split()) <= 6:
            trimmed_right = " ".join(right_words[:max_words])
            if len(right_words) > max_words:
                trimmed_right += "..."
            return f"{left}: {trimmed_right}"

    words = sentence.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."

    return sentence


def is_visual_or_flow_line(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    lowered = normalized.lower()

    if not normalized:
        return True

    visual_keywords = [
        "figure",
        "image",
        "diagram",
        "flowchart",
        "flow chart",
        "workflow",
        "process flow",
        "see figure",
        "refer to figure",
        "illustration",
        "smartart",
        "screenshot",
    ]

    if any(keyword in lowered for keyword in visual_keywords):
        return True

    if re.fullmatch(r"[>\-|=]{4,}", normalized):
        return True

    # Skip connector-heavy lines commonly produced by OCR on diagrams.
    if normalized.count("->") >= 1 or normalized.count("-->") >= 1:
        return True

    return False


def is_useful_keypoint(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip(" .:-|")
    lowered = normalized.lower()

    if len(normalized) < 12:
        return False

    if is_visual_or_flow_line(normalized):
        return False

    if re.fullmatch(r"[\W\d_]+", normalized):
        return False

    noise_patterns = [
        r"^page \d+$",
        r"^\d+$",
        r"^table of contents$",
    ]

    if any(re.fullmatch(pattern, lowered) for pattern in noise_patterns):
        return False

    if ESTIMATION_TIME_REGEX.search(normalized):
        return False

    return True


def split_table_cells(line: str) -> List[str]:
    if "|" in line:
        cells = [cell.strip(" :-") for cell in line.split("|")]
    elif "\t" in line:
        cells = [cell.strip(" :-") for cell in line.split("\t")]
    else:
        cells = re.split(r"\s{2,}", line)
        cells = [cell.strip(" :-") for cell in cells]

    return [
        cell for cell in cells
        if cell and not re.fullmatch(r"[-=]+", cell)
    ]


def normalize_heading_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip(" .:-|")
    normalized = re.sub(r"^\d+(\.\d+)*\s*", "", normalized)
    return normalized.lower()


def normalize_table_header_cell(text: str) -> str:
    normalized = normalize_heading_text(text)
    normalized = re.sub(r"[()_/\\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def detect_no_action_time_header(cells: List[str]) -> tuple[int, int] | None:
    if len(cells) < 3:
        return None

    normalized_cells = [normalize_table_header_cell(cell) for cell in cells]

    no_headers = {"no", "no.", "number", "seq", "sequence"}
    action_headers = {"action", "actions", "required action", "next action"}
    time_headers = {"time", "times", "duration", "eta", "estimation time"}

    for index in range(len(normalized_cells) - 2):
        first, second, third = normalized_cells[index:index + 3]
        if first in no_headers and second in action_headers and third in time_headers:
            return index + 1, index + 2

    return None


def is_table_header_row(cells: List[str]) -> bool:
    if len(cells) < 2:
        return False

    header_terms = {
        "step", "activity", "process", "task", "role", "owner",
        "action", "actions", "required action", "next action",
        "description", "details", "status", "remarks", "note", "notes",
        "no", "time",
    }
    lowered = [cell.strip().lower() for cell in cells if cell.strip()]
    if len(lowered) < 2:
        return False

    matches = sum(1 for cell in lowered if cell in header_terms)
    return matches >= max(1, min(2, len(lowered) - 1))


def extract_action_keypoints_from_table(text: str) -> List[str]:
    keypoints: List[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    action_headers = {"action", "actions", "required action", "next action"}
    context_headers = {"step", "activity", "process", "task", "role", "owner"}
    header_cells: List[str] = []
    action_index = -1
    time_index = -1
    no_action_time_mode = False

    for line in lines:
        if is_visual_or_flow_line(line):
            continue

        cells = split_table_cells(line)

        if len(cells) < 2:
            continue

        specific_header = detect_no_action_time_header(cells)
        if specific_header is not None:
            header_cells = [normalize_table_header_cell(cell) for cell in cells]
            action_index, time_index = specific_header
            no_action_time_mode = True
            continue

        lowered_cells = [normalize_table_header_cell(cell) for cell in cells]
        header_match = next(
            (idx for idx, cell in enumerate(lowered_cells) if cell in action_headers),
            -1
        )

        if header_match >= 0:
            header_cells = lowered_cells
            action_index = header_match
            time_index = next(
                (idx for idx, cell in enumerate(lowered_cells) if cell in {"time", "times", "duration", "eta", "estimation time"}),
                -1
            )
            no_action_time_mode = False
            continue

        if action_index < 0 or action_index >= len(cells):
            continue

        action_value = cells[action_index].strip()
        if not is_useful_keypoint(action_value):
            continue

        if no_action_time_mode:
            time_value = cells[time_index].strip() if 0 <= time_index < len(cells) else ""
            if time_value and is_useful_keypoint(f"Time {time_value}"):
                keypoints.append(f"{action_value} (Estimation time: {time_value})")
            else:
                keypoints.append(action_value)
            continue

        context_value = ""
        if header_cells:
            for idx, header in enumerate(header_cells):
                if idx == action_index or idx >= len(cells):
                    continue
                if header in context_headers and cells[idx].strip():
                    context_value = cells[idx].strip()
                    break

        time_value = cells[time_index].strip() if 0 <= time_index < len(cells) else ""
        candidate = f"{context_value}: {action_value}" if context_value else action_value
        if time_value and normalize_table_header_cell(time_value) not in {"time", "times", "duration", "eta", "estimation time"}:
            candidate = f"{candidate} (Estimation time: {time_value})"
        if is_useful_keypoint(candidate):
            keypoints.append(candidate)

    return keypoints


def extract_action_focused_fallback_from_table(text: str) -> List[str]:
    fallback_keypoints: List[str] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        if is_visual_or_flow_line(line):
            continue

        cells = split_table_cells(line)

        if len(cells) < 2 or is_table_header_row(cells):
            continue

        best_cell = ""
        best_score = float("-inf")
        for cell in cells:
            if not is_useful_keypoint(cell):
                continue
            score = sentence_information_score(cell)
            if score > best_score:
                best_score = score
                best_cell = cell.strip()

        if best_cell:
            fallback_keypoints.append(best_cell)

    return fallback_keypoints

def extract_table_keypoints(text: str) -> List[str]:
    keypoints = extract_action_keypoints_from_table(text)
    if keypoints:
        return keypoints

    return extract_action_focused_fallback_from_table(text)


def prepare_keypoint_source(text: str, section_title: str | None = None) -> str:
    if not text:
        return ""

    text = strip_estimation_time(text)
    text = re.sub(r"^\d+(\.\d+)*\s*", "", text, flags=re.MULTILINE)

    normalized_section_title = normalize_heading_text(section_title) if section_title else ""

    if section_title:
        text = re.sub(
            rf"^{re.escape(section_title)}[\s:.-]*",
            "",
            text,
            flags=re.IGNORECASE | re.MULTILINE
        )

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line or is_visual_or_flow_line(line):
            continue
        if normalized_section_title and normalize_heading_text(line) == normalized_section_title:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ============================================================
# KEYPOINT EXTRACTION
# ============================================================
def extract_keypoints(
    text: str,
    section_title: str | None = None,
    question: str | None = None,
) -> List[str]:

    if not text:
        return []

    prepared_text = prepare_keypoint_source(text, section_title)
    if not prepared_text:
        return []

    query_terms = extract_query_terms(question or "")
    table_keypoints = extract_table_keypoints(prepared_text)
    candidate_points: List[str] = []

    if table_keypoints:
        candidate_points.extend(table_keypoints)

    # Preserve explicit bullet formatting when present in the source text.
    explicit_bullets = re.findall(
        r"(?m)^\s*[\-•]\s+(.+?)\s*$",
        prepared_text
    )
    if explicit_bullets and not candidate_points:
        for item in explicit_bullets:
            candidate_points.extend(split_long_statement(item))

    text = re.sub(r"\s+", " ", prepared_text).strip()

    if not candidate_points:
        sentence_candidates = split_into_sentences(text)
        informative_sentences = [
            sentence for sentence in sentence_candidates
            if is_useful_keypoint(sentence)
        ]
        if informative_sentences:
            candidate_points.extend(informative_sentences)

    if not candidate_points:
        # Split around common policy connectors to avoid one very long bullet.
        connector_patterns = [
            r"(?i)\.\s+(?=Employees?\b)",
            r"(?i)\.\s+(?=The company\b)",
            r"(?i)\.\s+(?=Sick days?\b)",
            r"(?i)\.\s+(?=Abuse of this policy\b)",
            r"(?i)\.\s+(?=Managers?\b)",
            r"(?i)\.\s+(?=Upon\b)",
            r"(?i)\.\s+(?=If\b)",
        ]

        segmented = [text]
        for pattern in connector_patterns:
            updated = []
            for segment in segmented:
                updated.extend(re.split(pattern, segment))
            segmented = updated

        for segment in segmented:
            candidate_points.extend(split_long_statement(segment))

    cleaned_points = rank_candidate_points(candidate_points, query_terms)

    if cleaned_points:
        return cleaned_points[:MAX_KEYPOINTS]

    fallback = summarize_keypoint(text)
    return [fallback] if is_useful_keypoint(fallback) else []


def remove_first_keypoint(keypoints: List[str]) -> List[str]:
    if len(keypoints) <= 1:
        return []
    return keypoints[1:]

# ============================================================
# OVERVIEW GENERATION (Deterministic)
# ============================================================

def build_overview(keypoints: List[str]) -> str:
    if not keypoints:
        return ""

    first = keypoints[0]

    # Only first sentence
    sentence = first.split(".")[0]

    words = sentence.split()

    if len(words) > 20:
        return " ".join(words[:20]) + "..."

    return sentence.strip()


# ============================================================
# FORMAT OUTPUT (Compliance Mode)
# ============================================================

def format_section(title: str, raw_text: str, question: str | None = None) -> tuple[str, List[str], str]:
    """
    Return formatted answer AND extracted keypoints.
    """

    keypoints = extract_keypoints(raw_text, title, question)
    subsection_summaries = build_subsection_summaries(question or "", title, raw_text)
    summary = build_keyword_summary(question or "", title, raw_text, keypoints)
    if subsection_summaries:
        top_subsection = subsection_summaries[0]
        keypoints = top_subsection.get("keypoints") or keypoints
        summary = (top_subsection.get("summary") or "").strip() or summary
    resume_points = build_resume_points_from_subsections(
        question or "",
        subsection_summaries,
        raw_text,
        keypoints,
    )

    output = []

    output.append("Key Points")
    output.append("")

    if resume_points:
        for kp in resume_points:
            output.append(f"- {kp}")
    else:
        cleaned = clean_text(raw_text)
        output.append(cleaned if cleaned else "No key points available.")

    return "\n".join(output), resume_points, summary


# ============================================================
# MAIN CHAIN
# ============================================================

def get_chain(workspace_id: str):

    def run(question: str, history=None):

        history = history or []

        docs = retrieve_documents(workspace_id, question)

        if not docs:
            return {
                "answer": NO_ANSWER,
                "sources": []
            }

        best, selected_title, selected_text = select_best_doc_match(docs, question)
        if best is None:
            return {
                "answer": NO_ANSWER,
                "sources": []
            }

        page = best.metadata.get("page", 0)

        combined_text, source_refs = collect_subsection_across_pages(
            workspace_id,
            best,
            selected_title,
            selected_text,
        )

        formatted_answer, keypoints, summary = format_section(selected_title, combined_text, question)

        highlight_seed_points = keypoints[:]
        if summary:
            highlight_seed_points.insert(0, summary)

        highlights = select_highlight_snippets(
            question,
            selected_title,
            combined_text,
            highlight_seed_points,
        )

        highlight_pages = rank_highlight_pages(
            source_refs,
            question,
            highlights,
            selected_title,
        )
        if not highlight_pages:
            highlight_pages = [page, page + 1, page + 2]

        return {
            "answer": formatted_answer,
            "highlight": highlights,
            "highlight_pages": highlight_pages,
            "sources": build_sources(source_refs),
            "selected_section_title": selected_title,
            "summary": summary,
        }

    return run
