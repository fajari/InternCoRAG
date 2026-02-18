import re
from typing import List
from langchain.schema import Document

SECTION_REGEX = re.compile(
    r"(?:^|\n)(\d+\.\d+)\s+([A-Z][^\n]+)",
    re.MULTILINE
)


TOC_ENTRY_REGEX = re.compile(
    r"(\d+\.\d+)\s+(.+?)\s+\.{2,}\s*\d+"
)

MIN_SECTION_LENGTH = 120


def extract_toc_entries(text: str) -> List[str]:
    entries = []
    for m in TOC_ENTRY_REGEX.finditer(text):
        title = f"{m.group(1)} {m.group(2)}".strip()
        entries.append(title)
    return entries


def is_table_of_contents(text: str) -> bool:
    t = text.lower()

    if "table of contents" in t:
        return True

    if re.search(r"\.{4,}\s*\d+$", t, re.MULTILINE):
        return True

    return False


def chunk_documents(docs: List[Document], original_filename: str) -> List[Document]:
    chunks: List[Document] = []
    global_toc_entries: List[str] = []

    # ---------------------------------------------
    # PASS 1 → Extract TOC
    # ---------------------------------------------
    for doc in docs:
        text = doc.page_content.strip()

        if is_table_of_contents(text):
            toc_entries = extract_toc_entries(text)
            global_toc_entries.extend(toc_entries)

    # ---------------------------------------------
    # PASS 2 → Create Chunks
    # ---------------------------------------------
    for doc in docs:
        page = doc.metadata.get("page", 0)
        text = doc.page_content.strip()
        text = remove_repeated_header(text)

        if page <= 1:
            continue

        if is_table_of_contents(text):
            continue

        if is_version_history(text):
            continue

        matches = list(SECTION_REGEX.finditer(text))

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
                        "toc_entries": global_toc_entries,
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
                        "toc_entries": global_toc_entries,
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
