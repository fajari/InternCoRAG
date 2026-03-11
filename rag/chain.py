import re
from typing import List
from rag.retriever import retrieve_documents
from rag.helpers import build_sources

NO_ANSWER = "I don't know based on the provided documents."


# ============================================================
# CLEANING UTILITIES
# ============================================================

def clean_text(text: str) -> str:
    """
    Clean numbering artifacts and normalize spacing.
    """

    if not text:
        return ""

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


# ============================================================
# KEYPOINT EXTRACTION
# ============================================================
def extract_keypoints(text: str, section_title: str | None = None) -> List[str]:

    if not text:
        return []

    # Remove section numbering like "8.3 Termination"
    text = re.sub(r"^\d+(\.\d+)*\s*", "", text, flags=re.MULTILINE)

    # Normalize spacing
    text = re.sub(r"\s+", " ", text).strip()

    # Remove duplicated section title inside text
    if section_title:
        text = re.sub(
            rf"^{re.escape(section_title)}[\s:.-]*",
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

    keypoints: List[str] = []

    # Preserve explicit bullet formatting when present in the source text.
    explicit_bullets = re.findall(r"(?:^|\s)[\-•]\s+(.+?)(?=(?:\s[\-•]\s+)|$)", text)
    if explicit_bullets:
        for item in explicit_bullets:
            keypoints.extend(split_long_statement(item))
        return remove_duplicate_sentences(keypoints)[:10]

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
        keypoints.extend(split_long_statement(segment))

    cleaned_points = remove_duplicate_sentences(keypoints)

    if cleaned_points:
        return cleaned_points[:10]

    return [text]

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

def format_section(title: str, raw_text: str) -> tuple[str, List[str]]:
    """
    Return formatted answer AND extracted keypoints.
    """

    cleaned = clean_text(raw_text)

    keypoints = extract_keypoints(cleaned, title)

    overview = build_overview(keypoints)

    output = []

    output.append(title)
    output.append("")
    output.append("Overview")
    output.append("")
    output.append(overview if overview else "No summary available.")
    output.append("")
    output.append("Key Points")
    output.append("")

    for kp in keypoints:
        output.append(f"- {kp}")

    return "\n".join(output), keypoints


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

        # Only best section
        best = docs[0]

        raw_text = best.page_content
        page = best.metadata.get("page", 0)
        title = best.metadata.get("section", "Policy Section")

        formatted_answer, keypoints = format_section(title, raw_text)

        # Highlight from top 2 keypoints
        highlights = keypoints[:2] if keypoints else []

        return {
            "answer": formatted_answer,
            "highlight": highlights,
            "highlight_pages": [page],
            "sources": build_sources([best])
        }

    return run
