import re
import math
from typing import List, Dict, Any
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP
from narrative.classifier import heuristic_classify
from rag.vectorstore import embeddings

SECTION_REGEX = re.compile(
    r"(?<!\w)((?:\d+\.)+\d+)\s+([A-Z][A-Za-z0-9/&()' -]{2,120}?)"
    r"(?=\s+(?:Estimation time in total:|NO ACTION|$))"
)


TOC_ENTRY_REGEX = re.compile(
    r"((?:\d+\.)+\d+)\s+(.+?)\s+\.{2,}\s*(\d+)"
    r"(?=\s+(?:(?:\d+\.)+\d+)\s+|$)"
)

MIN_SECTION_LENGTH = 120
SEMANTIC_MIN_CHUNK_CHARS = max(320, CHUNK_SIZE // 2)
SEMANTIC_MAX_CHUNK_CHARS = max(900, int(CHUNK_SIZE * 1.8))
SEMANTIC_UNIT_TARGET_CHARS = 420
SEMANTIC_SIMILARITY_THRESHOLD = 0.42

NARRATIVE_TERMS = {
    "bahwa", "peristiwa", "kejadian", "kronologi", "kemudian", "selanjutnya",
    "setelah", "sebelum", "pada tanggal", "dilakukan", "menerima", "mengirim",
    "mentransfer", "membayar", "melaporkan", "menemui", "menghubungi",
    "terdakwa", "korban", "saksi", "pemohon", "termohon", "penggugat",
    "tergugat", "the victim", "witness", "reported", "transferred",
}
EVIDENCE_TERMS = {
    "bukti", "barang bukti", "surat", "lampiran", "rekening", "mutasi",
    "invoice", "kwitansi", "nota", "kontrak", "rekaman", "screenshot",
    "foto", "dokumen", "putusan", "berita acara", "hasil pemeriksaan",
    "exhibit", "evidence", "attachment", "appendix",
}
METADATA_TERMS = {
    "nomor perkara", "putusan nomor", "identitas", "nama", "tempat lahir",
    "umur", "tanggal lahir", "jenis kelamin", "kebangsaan", "tempat tinggal",
    "agama", "pekerjaan", "document number", "classification", "status",
    "approved by", "effective date", "author", "owner",
}


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


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def normalize_chunk_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def split_sentences(text: str) -> List[str]:
    normalized = normalize_chunk_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|;\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def pack_sentences(sentences: List[str], target_chars: int) -> List[str]:
    units: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        projected = current_len + len(sentence) + (1 if current else 0)
        if current and projected > target_chars:
            units.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len = projected

    if current:
        units.append(" ".join(current).strip())

    return units


def split_semantic_units(text: str) -> List[str]:
    if not text:
        return []

    paragraph_candidates = [
        normalize_chunk_text(part)
        for part in re.split(r"\n\s*\n+", text)
    ]
    paragraph_candidates = [part for part in paragraph_candidates if part]

    if not paragraph_candidates:
        paragraph_candidates = [normalize_chunk_text(text)]

    units: List[str] = []
    for paragraph in paragraph_candidates:
        if len(paragraph) <= SEMANTIC_UNIT_TARGET_CHARS:
            units.append(paragraph)
            continue

        sentence_units = pack_sentences(
            split_sentences(paragraph),
            SEMANTIC_UNIT_TARGET_CHARS,
        )
        units.extend(sentence_units or [paragraph])

    return [unit for unit in units if unit]


def build_overlap_prefix(text: str, max_chars: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    selected: List[str] = []
    total_len = 0
    for sentence in reversed(sentences):
        projected = total_len + len(sentence) + (1 if selected else 0)
        if selected and projected > max_chars:
            break
        selected.insert(0, sentence)
        total_len = projected

    return " ".join(selected).strip()


def semantic_subchunks(text: str) -> List[str]:
    normalized = text.strip()
    if len(normalized) <= CHUNK_SIZE:
        return [normalized]

    units = split_semantic_units(normalized)
    if len(units) <= 1:
        return [normalized]

    try:
        vectors = embeddings.embed_documents(units)
    except Exception:
        return [normalized]

    chunks: List[str] = []
    current_units = [units[0]]
    current_vector = vectors[0]
    current_len = len(units[0])

    for unit, vector in zip(units[1:], vectors[1:]):
        similarity = cosine_similarity(current_vector, vector)
        projected_len = current_len + 2 + len(unit)

        should_merge = (
            current_len < SEMANTIC_MIN_CHUNK_CHARS
            or projected_len <= CHUNK_SIZE
            or (similarity >= SEMANTIC_SIMILARITY_THRESHOLD and projected_len <= SEMANTIC_MAX_CHUNK_CHARS)
            or len(unit) < 140
        )

        if should_merge:
            current_units.append(unit)
            current_len = projected_len
            current_vector = [
                (a + b) / 2.0
                for a, b in zip(current_vector, vector)
            ]
            continue

        chunks.append("\n\n".join(current_units).strip())
        current_units = [unit]
        current_vector = vector
        current_len = len(unit)

    if current_units:
        chunks.append("\n\n".join(current_units).strip())

    if len(chunks) <= 1:
        return chunks

    overlapped_chunks: List[str] = []
    previous_text = ""
    for chunk in chunks:
        if previous_text and CHUNK_OVERLAP > 0:
            overlap = build_overlap_prefix(previous_text, CHUNK_OVERLAP)
            if overlap and not chunk.startswith(overlap):
                chunk = f"{overlap}\n\n{chunk}".strip()
        overlapped_chunks.append(chunk)
        previous_text = chunk

    return overlapped_chunks


def count_keyword_hits(text: str, keywords: set[str]) -> int:
    lowered = (text or "").lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def looks_like_table_or_attachment(text: str) -> bool:
    lines = [line for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return False

    tableish_lines = 0
    for line in lines:
        stripped = line.strip()
        if "|" in stripped or "\t" in stripped:
            tableish_lines += 1
            continue
        if len(re.split(r"\s{2,}", stripped)) >= 3:
            tableish_lines += 1

    return tableish_lines >= max(2, len(lines) // 3)


def classify_document_part(text: str) -> str:
    return heuristic_classify(text).classification


def build_chunk_documents(text: str, metadata: Dict[str, Any]) -> List[Document]:
    subchunks = semantic_subchunks(text)
    total = len(subchunks)
    built: List[Document] = []

    for index, chunk_text in enumerate(subchunks):
        classification = heuristic_classify(chunk_text)
        content_type = classification.classification
        chunk_id = (
            f"{re.sub(r'[^A-Za-z0-9]+', '-', str(metadata.get('source', 'doc'))).strip('-').lower() or 'doc'}"
            f"-p{int(metadata.get('page', 0)) + 1}-c{index}"
        )
        built.append(
            Document(
                page_content=chunk_text,
                metadata={
                    **metadata,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "classification": content_type,
                    "classification_confidence": round(classification.confidence, 3),
                    "content_type": content_type,
                    "is_narrative": content_type == "narrative",
                    "semantic_chunk_index": index,
                    "semantic_chunk_total": total,
                }
            )
        )

    return built


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

        if is_table_of_contents(text):
            continue

        # Without a TOC, keep the previous defensive cover-page skip.
        if first_toc_page is None and page <= 1 and is_front_matter(text):
            continue

        matches = extract_section_matches(text)

        # NO SECTION → WHOLE PAGE
        if not matches:
            if len(text) < 300:
                continue

            chunks.extend(
                build_chunk_documents(
                    text,
                    {
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

            chunks.extend(
                build_chunk_documents(
                    section_text,
                    {
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
