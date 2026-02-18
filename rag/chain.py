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
            re.escape(section_title),
            "",
            text,
            flags=re.IGNORECASE
        ).strip()

    keypoints = []

    # ---------------------------------------------------
    # STEP 1: Split intro vs list (detect "required")
    # ---------------------------------------------------

    match = re.search(r"\brequired\b", text, re.IGNORECASE)

    if match:
        intro = text[:match.end()].strip()
        remainder = text[match.end():].strip()
    else:
        return [text]

    # Clean intro
    intro = intro.strip(" .:-")
    if len(intro) > 25:
        keypoints.append(intro)

    # ---------------------------------------------------
    # STEP 2: Extract dash list items
    # ---------------------------------------------------

    # Find all occurrences of "- to ..."
    items = re.findall(r"-\s*to\s+([^.;]+)", remainder, re.IGNORECASE)

    for item in items:
        item = item.strip(" .:-")
        if len(item) > 15:
            keypoints.append("To " + item)

    # ---------------------------------------------------
    # STEP 3: Remove duplicates safely
    # ---------------------------------------------------

    seen = set()
    clean_points = []

    for kp in keypoints:
        k = kp.lower()
        if k not in seen:
            seen.add(k)
            clean_points.append(kp)

    return clean_points


    # -----------------------------------------
    # STEP 1: Split intro vs detailed list
    # -----------------------------------------

    # Detect "Upon termination" as logical split
    split_marker = re.search(r"\bUpon\b", text, re.IGNORECASE)

    if split_marker:
        intro = text[:split_marker.start()].strip()
        rest = text[split_marker.start():].strip()
    else:
        # fallback
        parts = text.split(":", 1)
        if len(parts) == 2:
            intro, rest = parts
        else:
            intro = text
            rest = ""

    # -----------------------------------------
    # STEP 2: Clean intro
    # -----------------------------------------

    intro = intro.strip(" .:-")

    if len(intro) > 30:
        keypoints.append(intro)

    # -----------------------------------------
    # STEP 3: Extract bullet-style items
    # -----------------------------------------

    if rest:

        # split by dash bullets or semicolon
        items = re.split(r"\s-\s|;\s*", rest)

        for item in items:
            item = item.strip(" .:-")

            if not item:
                continue

            # remove duplicate intro
            if item.lower() == intro.lower():
                continue

            # skip garbage short lines
            if len(item) < 15:
                continue

            keypoints.append(item)

    # -----------------------------------------
    # STEP 4: Remove duplicates cleanly
    # -----------------------------------------

    seen = set()
    final = []

    for kp in keypoints:
        key = kp.lower()
        if key not in seen:
            seen.add(key)
            final.append(kp)

    return final[:10]

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
        output.append(f"• {kp}")

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
