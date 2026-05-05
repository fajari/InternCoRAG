import re
from typing import List, Dict, Any
from langchain.schema import Document

SECTION_REGEX = re.compile(
    r"(?<!\w)((?:\d+\.)+\d+)\s+([A-Z][A-Za-z0-9/&()' -]{2,120}?)"
    r"(?=\s+(?:Estimation time in total:|NO ACTION|$))"
)


TOC_ENTRY_REGEX = re.compile(
    r"((?:\d+\.)+\d+)\s+(.+?)\s+\.{2,}\s*(\d+)"
    r"(?=\s+(?:(?:\d+\.)+\d+)\s+|$)"
)

MIN_SECTION_LENGTH = 120


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", normalize_text(text))


def extract_section_number(text: str) -> str:
    match = re.match(r"^\s*(((?:\d+\.)*\d+))\b", text or "")
    return match.group(1) if match else ""


def is_same_number_lineage(section_number: str, entry_number: str) -> bool:
    if not section_number or not entry_number:
        return False

    return (
        entry_number == section_number
        or entry_number.startswith(f"{section_number}.")
        or section_number.startswith(f"{entry_number}.")
    )


def text_tokens_without_numbers(text: str) -> set[str]:
    return {token for token in tokenize(text) if not token.isdigit()}


def extract_toc_entries(text: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for m in TOC_ENTRY_REGEX.finditer(text):
        number = m.group(1).strip()
        heading = re.sub(r"\s+", " ", m.group(2)).strip()
        full_title = f"{number} {heading}".strip()
        entries.append({
            "number": number,
            "title": heading,
            "full_title": full_title,
            "page_ref": int(m.group(3)),
        })
    return entries


def extract_section_matches(text: str) -> List[re.Match]:
    return list(SECTION_REGEX.finditer(text or ""))


def dedupe_toc_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_entries: List[Dict[str, Any]] = []

    for entry in entries:
        key = (
            entry.get("number", ""),
            normalize_text(entry.get("title", "")),
            entry.get("page_ref"),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_entries.append(entry)

    return unique_entries


def select_relevant_toc_entries(
    toc_entries: List[Dict[str, Any]],
    section_title: str,
    page: int,
) -> List[str]:
    if not toc_entries:
        return []

    section_number = extract_section_number(section_title)
    section_tokens = text_tokens_without_numbers(section_title)
    relevant: List[tuple[int, str]] = []

    for entry in toc_entries:
        entry_number = entry.get("number", "")
        entry_full_title = entry.get("full_title", "").strip()
        entry_page_ref = entry.get("page_ref")
        entry_tokens = text_tokens_without_numbers(entry_full_title)

        score = 0
        same_lineage = is_same_number_lineage(section_number, entry_number)

        if section_number and entry_number == section_number:
            score += 12
        elif same_lineage:
            score += 8

        if same_lineage or not section_number:
            overlap = len(section_tokens & entry_tokens)
            if overlap:
                score += overlap * 3

        if same_lineage and isinstance(entry_page_ref, int) and abs(entry_page_ref - (page + 1)) <= 1:
            score += 4

        if score > 0 and entry_full_title:
            relevant.append((score, entry_full_title))

    relevant.sort(key=lambda item: item[0], reverse=True)

    selected: List[str] = []
    seen_titles = set()
    for _, title in relevant:
        title_norm = normalize_text(title)
        if title_norm in seen_titles:
            continue
        seen_titles.add(title_norm)
        selected.append(title)
        if len(selected) >= 4:
            break

    return selected


def is_table_of_contents(text: str) -> bool:
    t = text.lower()

    if "table of contents" in t:
        return True

    if re.search(r"(?m)^\s*toc\s*$", t):
        return True

    if re.search(r"\.{4,}\s*\d+$", t, re.MULTILINE):
        return True

    return False


def is_front_matter(text: str) -> bool:
    t = text.lower()

    keyword_groups = [
        ("authorship", "approval"),
        ("author", "approved by"),
        ("classification", "status"),
        ("document number", "reference code"),
        ("document no", "reference code"),
        ("document number", "document status"),
    ]

    if any(all(keyword in t for keyword in group) for group in keyword_groups):
        return True

    metadata_keywords = [
        "authorship",
        "approval",
        "approved by",
        "classification",
        "document status",
        "status",
        "document number",
        "document no",
        "reference code",
        "reference no",
        "effective date",
        "owner",
    ]

    matched_keywords = sum(1 for keyword in metadata_keywords if keyword in t)

    # Front-matter pages are usually short and metadata-dense.
    return matched_keywords >= 3 and len(t) < 2500


def chunk_documents(docs: List[Document], original_filename: str) -> List[Document]:
    chunks: List[Document] = []
    global_toc_entries: List[Dict[str, Any]] = []
    toc_pages = set()

    # ---------------------------------------------
    # PASS 1 → Extract TOC
    # ---------------------------------------------
    for doc in docs:
        text = doc.page_content.strip()

        if is_table_of_contents(text):
            toc_entries = extract_toc_entries(text)
            global_toc_entries.extend(toc_entries)
            toc_pages.add(doc.metadata.get("page", 0))

    global_toc_entries = dedupe_toc_entries(global_toc_entries)
    first_toc_page = min(toc_pages) if toc_pages else None

    # ---------------------------------------------
    # PASS 2 → Create Chunks
    # ---------------------------------------------
    for doc in docs:
        page = doc.metadata.get("page", 0)
        text = doc.page_content.strip()
        text = remove_repeated_header(text)

        # If a TOC exists, treat everything before it as front matter.
        if first_toc_page is not None and page < first_toc_page:
            continue

        if is_table_of_contents(text):
            continue

        # Without a TOC, keep the previous defensive cover-page skip.
        if first_toc_page is None and page <= 1:
            continue

        # Skip metadata-heavy front matter pages even if they appear
        # around the TOC area before the real document sections start.
        if is_front_matter(text):
            continue

        if is_version_history(text):
            continue

        matches = extract_section_matches(text)

        # NO SECTION → WHOLE PAGE
        if not matches:
            if len(text) < 300:
                continue

            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        **doc.metadata,
                        "section": f"General (page {page})",
                        "page": page,
                        "source": original_filename,
                        "toc_entries": select_relevant_toc_entries(
                            global_toc_entries,
                            f"General (page {page})",
                            page,
                        ),
                    }
                )
            )
            continue

        # SPLIT BY SECTION
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            section_text = text[start:end].strip()

            if len(section_text) < MIN_SECTION_LENGTH:
                continue

            section_title = f"{m.group(1)} {m.group(2)}".strip()

            chunks.append(
                Document(
                    page_content=section_text,
                    metadata={
                        **doc.metadata,
                        "section": section_title,
                        "page": page,
                        "source": original_filename,
                        "toc_entries": select_relevant_toc_entries(
                            global_toc_entries,
                            section_title,
                            page,
                        ),
                    }
                )
            )

    return chunks

def is_version_history(text: str) -> bool:
    t = text.lower()

    keywords = [
        "version history",
        "revision history",
        "document control",
        "change log",
        "revision log",
        "amendment history",
        "version control",
        "approved by",
        "effective date"
    ]

    if any(k in t for k in keywords):
        return True

    # detect revision table style
    if "version" in t and "date" in t and "description" in t:
        return True

    return False

def remove_repeated_header(text: str) -> str:
    lines = text.splitlines()

    if len(lines) < 3:
        return text

    # ambil 1–3 baris paling atas (biasanya header)
    top_lines = lines[:3]

    cleaned_lines = []
    for line in lines:
        # skip line kalau mirip header
        if any(
            header_line.strip() and
            header_line.strip() in line
            for header_line in top_lines
        ):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
