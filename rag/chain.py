import re
from typing import Any, List
from qdrant_client.http import models
from config import QDRANT_COLLECTION
from narrative import analyze_documents_sync
from narrative.extractor import top_actors
from rag.graph_pipeline import build_graph_bundle
from rag.query import QueryAnalysis, build_query_analysis, canonicalize_token, tokenize_question
from rag.retriever import retrieve_documents
from rag.helpers import build_sources
from rag.vectorstore import client

NO_ANSWER = "I don't know based on the provided documents."
DISABLED_MODE_ANSWER = (
    "Mode Analisis Isi PDF sementara dinonaktifkan. "
    "Silakan gunakan Diagram Jaringan Sosial atau ajukan pertanyaan spesifik lewat chat lanjutan."
)
DISABLED_RESPONSE_MODES = {"analysis"}
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
CASE_SUMMARY_KEYWORDS = (
    "ringkasan",
    "rangkuman",
    "resume",
    "summary",
    "summarize",
)
CASE_ANALYSIS_KEYWORDS = (
    "analisa",
    "analisis",
    "analyze",
    "analysis",
    "review kasus",
    "bedah kasus",
)
SOCIAL_GRAPH_KEYWORDS = (
    "diagram jaringan sosial",
    "jaringan sosial",
    "social network",
    "relationship map",
    "relasi pihak",
    "hubungan antar pihak",
    "pihak terlibat",
)
TIMELINE_EVENT_KEYWORDS = (
    "pada", "tanggal", "kemudian", "selanjutnya", "setelah itu", "setelah",
    "sebelumnya", "awalnya", "akhirnya", "lalu", "saat", "ketika",
    "dilaporkan", "ditransfer", "dibayar", "diperiksa", "diinvestigasi",
    "dipanggil", "disetujui", "ditolak", "menerima", "mengirim",
)
TIMELINE_MARKER_REGEX = re.compile(
    r"(?i)\b("
    r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"\d{1,2}\s+"
    r"(?:januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{2,4}|"
    r"(?:januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{4}|"
    r"hari\s+(?:yang\s+)?sama|hari\s+berikutnya|minggu\s+berikutnya|bulan\s+berikutnya"
    r")\b"
)
ACTOR_STOPWORDS = {
    "The",
    "This",
    "That",
    "These",
    "Those",
    "Page",
    "Case",
    "Document",
    "Summary",
    "Ringkasan",
    "Analisis",
    "Diagram",
}
ACTOR_GENERIC_TERMS = {
    "kasus", "dokumen", "laporan", "ringkasan", "analisis", "diagram",
    "halaman", "page", "pages", "pdf", "ocr", "summary", "document",
    "case", "company", "internal", "knowledge", "assistant",
}
ACTOR_ROLE_TERMS = {
    "korban", "pelaku", "tersangka", "terdakwa", "saksi", "penyidik", "jaksa",
    "hakim", "pengacara", "kuasa hukum", "perusahaan", "bank", "instansi",
    "direktur", "manajer", "pegawai", "karyawan", "nasabah", "klien",
}
ACTOR_LEAD_BLACKLIST = {
    "dan", "atau", "yang", "dengan", "pada", "untuk", "karena", "setelah",
    "sebelum", "dalam", "oleh",
}
FORENSIC_ROLE_PATTERNS = [
    ("Korban", (r"(?i)\bkorban\b|\bvictim\b|\bpihak yang dirugikan\b",)),
    ("Pelaku", (r"(?i)\bpelaku\b|\btersangka\b|\bterdakwa\b|\bsuspect\b|\bactor\b",)),
    ("Saksi", (r"(?i)\bsaksi\b|\bwitness\b",)),
    ("Penyidik", (r"(?i)\bpenyidik\b|\binvestigator\b|\bpolisi\b|\bpolice\b",)),
    ("Penegak Hukum", (r"(?i)\bjaksa\b|\bprosecutor\b|\bhakim\b|\bjudge\b",)),
    ("Kuasa Hukum", (r"(?i)\bpengacara\b|\blawyer\b|\badvokat\b|\bkuasa hukum\b",)),
    ("Instansi", (r"(?i)\bperusahaan\b|\bbank\b|\binstansi\b|\bpt\b|\bcv\b|\bkantor\b|\bagency\b",)),
]
ROLE_DISPLAY_ORDER = [
    "Pelaku",
    "Korban",
    "Saksi",
    "Instansi",
    "Pihak Lain",
    "Penyidik",
    "Penegak Hukum",
    "Kuasa Hukum",
]
CASE_CORE_ROLES = {"Pelaku", "Korban", "Saksi", "Instansi", "Pihak Lain"}
CASE_SUPPORT_ROLES = {"Penyidik", "Penegak Hukum", "Kuasa Hukum"}
RELATION_PATTERNS = [
    (r"(?i)\bmelaporkan\b|\breported\b|\breporting\b", "melaporkan"),
    (r"(?i)\bmenuduh\b|\baccused\b|\baccusing\b", "menuduh"),
    (r"(?i)\bmenghubungi\b|\bcontacted\b|\bcontacting\b", "menghubungi"),
    (r"(?i)\bmentransfer\b|\btransfer(?:red)?\b|\bsent\b", "mentransfer"),
    (r"(?i)\bmembayar\b|\bpaid\b|\bpayment\b", "membayar"),
    (r"(?i)\bmenerima\b|\breceived\b|\baccept(?:ed)?\b", "menerima"),
    (r"(?i)\bmemeriksa\b|\binvestigat(?:e|ed)\b|\breviewed\b", "memeriksa"),
    (r"(?i)\bmenyetujui\b|\bapproved\b|\bauthorized\b", "menyetujui"),
    (r"(?i)\bmemerintahkan\b|\binstructed\b|\bordered\b", "memerintahkan"),
    (r"(?i)\bbertemu\b|\bmet\b|\bmeeting\b", "bertemu"),
    (r"(?i)\bberkomunikasi\b|\bcommunicat(?:ed|ion)\b", "berkomunikasi"),
]
FLOW_RELATION_LABELS = {"mentransfer", "membayar", "menerima"}
COMMUNICATION_RELATION_LABELS = {"menghubungi", "bertemu", "berkomunikasi"}
LEGAL_RELATION_LABELS = {"melaporkan", "menuduh", "memeriksa", "menyetujui", "memerintahkan"}
STRONG_EVIDENCE_TERMS = {
    "bukti", "rekening", "transfer", "invoice", "kwitansi", "dokumen",
    "rekaman", "laporan", "kontrak", "surat", "putusan", "pemeriksaan",
    "investigasi", "approved", "authorized",
}
WEAK_EVIDENCE_TERMS = {
    "dugaan", "diduga", "indikasi", "kemungkinan", "seolah", "rumor",
    "perkiraan", "mungkin", "didapati", "diasumsikan",
}
LEGAL_BOILERPLATE_PATTERNS = [
    re.compile(r"(?i)\bE-?mail\s*:\s*\S+"),
    re.compile(r"(?i)\bTelp\.?\s*:?\s*[\d()\-\s]+(?:ext\.?\s*\d+)?"),
    re.compile(r"(?i)\bFax\.?\s*:?\s*[\d()\-\s]+"),
    re.compile(r"(?i)\bwww\.\S+"),
    re.compile(r"(?i)\b(?:https?://)?putusan\.mahkamahagung\.go\.id\b"),
    re.compile(r"(?i)\bHalaman\s+\d+\s+dari\s+\d+\s+halaman\b"),
    re.compile(r"(?i)\bHalaman\s+\d+\b"),
    re.compile(r"(?i)\bPutusan\s+Nomor\s+[A-Za-z0-9./\- ]+\b"),
    re.compile(r"(?i)\bNomor\s+\d+\s+[A-Za-z./-]+\b"),
    re.compile(r"(?i)\bUntuk\s+salinan\s+yang\s+sama\s+bunyinya\b"),
    re.compile(r"(?i)\bPanitera\s+Pengganti\b"),
    re.compile(r"(?i)\bMajelis\s+Hakim\b"),
    re.compile(r"(?i)\bHakim\s+Anggota\b"),
    re.compile(r"(?i)\bHakim\s+Ketua\b"),
    re.compile(r"(?i)\bDemi\s+Keadilan\s+Berdasarkan\s+Ketuhanan\s+Yang\s+Maha\s+Esa\b"),
    re.compile(r"(?i)\bM E N G A D I L I\b"),
    re.compile(r"(?i)\bM E N I M B A N G\b"),
    re.compile(r"(?i)\bA m a r\b"),
    re.compile(
        r"(?i)\bDalam\s+hal\s+Anda\s+menemukan\s+inakurasi\s+informasi\s+yang\s+termuat\s+"
        r"pada\s+situs\s+ini.*?Kepaniteraan\s+Mahkamah\s+Agung\s+RI\s+melalui\b.*"
    ),
    re.compile(
        r"(?i)\bDokumen\s+Kasus\s+Dalam\s+hal\s+Anda\s+menemukan\s+inakurasi\s+informasi\b.*"
    ),
    re.compile(
        r"(?i)\bDalam\s+hal\s+Anda\s+menemukan\s+inakurasi\s+informasi\b.*?\bbelum\s+tersedia\b.*"
    ),
    re.compile(r"(?i)\bdisop\s*i\b.*"),
    re.compile(r"(?i)\bDensus\s+Ant\b.*"),
]
LEGAL_BOILERPLATE_LINE_PATTERNS = [
    re.compile(r"(?i)^\s*(e-?mail|telp|fax|halaman|putusan nomor|putusan\.mahkamahagung\.go\.id)\b"),
    re.compile(r"(?i)^\s*mahkamah agung\b"),
    re.compile(r"(?i)^\s*(majelis hakim|hakim ketua|hakim anggota|panitera pengganti|panitera)\b"),
    re.compile(r"(?i)^\s*(demi keadilan berdasarkan ketuhanan yang maha esa|m e n g a d i l i|m e n i m b a n g|amar putusan)\b"),
    re.compile(r"(?i)^\s*(untuk salinan yang sama bunyinya|kepaniteraan)\b"),
    re.compile(r"(?i)^\s*dalam\s+hal\s+anda\s+menemukan\s+inakurasi\s+informasi\b"),
    re.compile(r"(?i)^\s*dokumen\s+kasus\s+dalam\s+hal\s+anda\s+menemukan\s+inakurasi\s+informasi\b"),
    re.compile(r"(?i)^\s*(disop\s*i|densus\s+ant)\b"),
]
MAP_REDUCE_MAX_CHUNKS = 18
MAP_REDUCE_CHUNK_TARGET = 1400
IDENTITY_FIELDS = [
    "Nama",
    "Tempat Lahir",
    "Umur / Tanggal Lahir",
    "Jenis Kelamin",
    "Kebangsaan",
    "Tempat Tinggal",
    "Agama",
    "Pekerjaan",
]


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


def strip_legal_boilerplate(text: str) -> str:
    if not text:
        return ""

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if any(pattern.search(line) for pattern in LEGAL_BOILERPLATE_LINE_PATTERNS):
            continue
        if sum(1 for pattern in LEGAL_BOILERPLATE_PATTERNS if pattern.search(line)) >= 2:
            continue

        updated = line
        for pattern in LEGAL_BOILERPLATE_PATTERNS:
            updated = pattern.sub("", updated)
        updated = re.sub(r"\s{2,}", " ", updated).strip(" ,;:-")
        if updated:
            cleaned_lines.append(updated)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def normalize_formal_statement(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    if not normalized:
        return ""

    normalized = normalized[0].upper() + normalized[1:] if len(normalized) > 1 else normalized.upper()
    if normalized[-1] not in ".!?":
        normalized += "."
    return normalized


def join_formal_clauses(parts: List[str]) -> str:
    cleaned = [part.strip(" ,;:-") for part in parts if part and part.strip(" ,;:-")]
    if not cleaned:
        return ""
    statement = ", ".join(cleaned)
    return normalize_formal_statement(statement)

def clean_text(text: str) -> str:
    """
    Clean numbering artifacts and normalize spacing.
    """

    if not text:
        return ""

    text = strip_estimation_time(text)
    text = strip_legal_boilerplate(text)

    # Remove numbering like:
    # 8.3 Termination
    # 3 Termination
    text = re.sub(r"\n?\s*\d+(\.\d+)*\s+(?=[A-Z])", "\n", text)

    # Remove stray page number at end
    text = re.sub(r"\s\d+\s*$", "", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def build_ocr_tolerant_label_pattern(label: str) -> str:
    letters = [char for char in label.lower() if char.isalnum()]
    return r"[\s./:-]*".join(re.escape(char) for char in letters)


def normalize_identity_labels(text: str) -> str:
    normalized = text or ""
    for label in IDENTITY_FIELDS:
        pattern = build_ocr_tolerant_label_pattern(label)
        normalized = re.sub(
            rf"(?i){pattern}\s*:?",
            f"{label}:",
            normalized,
        )
    normalized = re.sub(r"\s{2,}", " ", normalized)
    return normalized


def extract_identity_source_text(source_refs: List, fallback_text: str) -> str:
    prioritized: List[tuple[int, str]] = []
    for item in source_refs:
        metadata = item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})
        content = item.get("page_content", "") if isinstance(item, dict) else getattr(item, "page_content", "")
        page = metadata.get("page", 0)
        prioritized.append((int(page) if isinstance(page, int) else 0, content or ""))

    prioritized.sort(key=lambda item: item[0])
    leading_parts = [text for _, text in prioritized[:4] if text.strip()]
    if leading_parts:
        return "\n".join(leading_parts)

    words = (fallback_text or "").split()
    return " ".join(words[:900])


def extract_petitioner_identity(source_refs: List, fallback_text: str) -> dict[str, str]:
    source_text = extract_identity_source_text(source_refs, fallback_text)
    if not source_text:
        return {}

    normalized = normalize_identity_labels(source_text)
    compacted = re.sub(r"\s+", " ", normalized).strip()
    if not compacted:
        return {}

    values: dict[str, str] = {}
    for index, label in enumerate(IDENTITY_FIELDS):
        next_labels = IDENTITY_FIELDS[index + 1:]
        next_pattern = "|".join(re.escape(next_label) + r"\s*:" for next_label in next_labels)
        if next_pattern:
            pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?=(?:{next_pattern})|$)"
        else:
            pattern = rf"{re.escape(label)}\s*:\s*(.*)$"

        match = re.search(pattern, compacted, flags=re.IGNORECASE)
        if not match:
            continue

        value = match.group(1).strip(" ,;:-")
        value = strip_legal_boilerplate(value)
        value = re.sub(r"\s+", " ", value).strip(" ,;:-")
        if not value:
            continue

        if len(value.split()) > 18:
            value = " ".join(value.split()[:18]).strip(" ,;:-")
        values[label] = normalize_formal_statement(value).rstrip(".")

    return values


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
        summary_parts.append(" ".join(summarize_keypoint(item, max_words=30) for item in selected))

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
    normalized = strip_legal_boilerplate(text or "")
    normalized = re.sub(r"\s+", " ", normalized).strip(" .:-")
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
        summary = summarize_keypoint(point, max_words=38)
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
        trimmed = words[:max_words]
        natural_break_tokens = {"dan", "atau", "karena", "yang", "sehingga", "untuk", "dengan"}
        for index in range(len(trimmed) - 1, max(0, len(trimmed) - 8), -1):
            if trimmed[index].lower().strip(",") in natural_break_tokens:
                trimmed = trimmed[:index]
                break
        return " ".join(trimmed).rstrip(",;:") + "..."

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

    if any(pattern.search(normalized) for pattern in LEGAL_BOILERPLATE_PATTERNS):
        pattern_hits = sum(1 for pattern in LEGAL_BOILERPLATE_PATTERNS if pattern.search(normalized))
        if pattern_hits >= 1 and len(normalized.split()) <= 18:
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
    text = strip_legal_boilerplate(text)
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


def detect_response_mode(question: str) -> str:
    normalized = re.sub(r"\s+", " ", (question or "").lower()).strip()

    if any(keyword in normalized for keyword in SOCIAL_GRAPH_KEYWORDS):
        return "social_graph"
    if any(keyword in normalized for keyword in CASE_ANALYSIS_KEYWORDS):
        return "analysis"
    if any(keyword in normalized for keyword in CASE_SUMMARY_KEYWORDS):
        return "summary"

    return "qa"


def collect_document_pages(
    workspace_id: str,
    source: str,
    fallback_docs: List,
) -> tuple[str, List]:
    source_docs = fetch_source_documents(workspace_id, source)
    if source_docs:
        combined = "\n".join(
            item.get("page_content", "").strip()
            for item in source_docs
            if item.get("page_content", "").strip()
        ).strip()
        return combined, source_docs

    combined = "\n".join(
        doc.page_content.strip()
        for doc in fallback_docs
        if getattr(doc, "page_content", "").strip()
    ).strip()
    return combined, fallback_docs


def collect_semantic_chunks(source_refs: List, fallback_text: str) -> List[str]:
    chunks: List[str] = []
    seen = set()

    for item in source_refs:
        content = item.get("page_content", "") if isinstance(item, dict) else getattr(item, "page_content", "")
        cleaned = prepare_keypoint_source(content)
        normalized = re.sub(r"\s+", " ", cleaned).strip().lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        chunks.append(cleaned)

    if chunks:
        return chunks[:MAP_REDUCE_MAX_CHUNKS]

    return pack_semantic_summary_chunks(fallback_text)


def get_source_ref_content(item: Any) -> str:
    return item.get("page_content", "") if isinstance(item, dict) else getattr(item, "page_content", "")


def get_source_ref_metadata(item: Any) -> dict:
    return item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})


def infer_content_type_from_text(text: str) -> str:
    lowered = (text or "").lower()
    metadata_hits = sum(1 for term in (
        "nomor perkara", "putusan nomor", "identitas", "nama:", "tempat lahir",
        "umur", "tanggal lahir", "jenis kelamin", "kebangsaan", "tempat tinggal",
        "agama", "pekerjaan", "document number", "classification", "approved by",
    ) if term in lowered)
    evidence_hits = sum(1 for term in STRONG_EVIDENCE_TERMS | {
        "barang bukti", "lampiran", "mutasi", "kwitansi", "nota", "screenshot",
        "foto", "berita acara", "exhibit", "attachment", "appendix",
    } if term in lowered)
    narrative_hits = sum(1 for term in set(TIMELINE_EVENT_KEYWORDS) | {
        "bahwa", "peristiwa", "kejadian", "kronologi", "terdakwa", "korban", "saksi",
        "pemohon", "termohon", "penggugat", "tergugat",
    } if term in lowered)
    tableish = any("|" in line or "\t" in line or len(re.split(r"\s{2,}", line.strip())) >= 3 for line in (text or "").splitlines())

    if metadata_hits >= 4 and narrative_hits <= 1:
        return "metadata"
    if (tableish and evidence_hits >= 1) or (evidence_hits >= 3 and narrative_hits <= 1):
        return "evidence"
    if narrative_hits >= 1 or TIMELINE_MARKER_REGEX.search(text or ""):
        return "narrative"
    if tableish:
        return "evidence"
    return "narrative"


def get_content_type(item: Any, content: str) -> str:
    metadata = get_source_ref_metadata(item)
    content_type = (metadata.get("content_type") or "").strip().lower()
    if content_type in {"narrative", "evidence", "metadata"}:
        return content_type
    return infer_content_type_from_text(content)


def extract_narrative_sentences(text: str) -> List[str]:
    sentences = []
    for sentence in split_into_sentences(text):
        lowered = sentence.lower()
        has_event = any(keyword in lowered for keyword in TIMELINE_EVENT_KEYWORDS)
        has_actor_role = any(role in lowered for role in ACTOR_ROLE_TERMS)
        has_date = bool(TIMELINE_MARKER_REGEX.search(sentence))
        if (has_event or has_actor_role or has_date) and is_useful_keypoint(sentence):
            sentences.append(sentence)
    return remove_duplicate_sentences(sentences)


def format_source_location(metadata: dict) -> str:
    source = metadata.get("source", "dokumen")
    page = metadata.get("page")
    if isinstance(page, int):
        return f"{source} hal. {page + 1}"
    return str(source)


def build_evidence_reference_lines(source_refs: List[Any], limit: int = 6) -> List[str]:
    candidates: List[tuple[float, str]] = []
    seen = set()

    for item in source_refs:
        content = prepare_keypoint_source(get_source_ref_content(item))
        metadata = get_source_ref_metadata(item)
        content_type = get_content_type(item, content)
        if content_type not in {"evidence", "narrative"}:
            continue
        for sentence in split_into_sentences(content):
            lowered = sentence.lower()
            evidence_hits = sum(1 for term in STRONG_EVIDENCE_TERMS if term in lowered)
            if content_type != "evidence" and evidence_hits == 0:
                continue
            detail = summarize_keypoint(sentence, max_words=28)
            if not detail:
                continue
            key = detail.lower()
            if key in seen:
                continue
            seen.add(key)
            score = evidence_hits * 3 + sentence_information_score(sentence)
            candidates.append((score, f"- {format_source_location(metadata)}: {detail}"))

    candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [line for _, line in candidates[:limit]]


def collect_case_reading(source_refs: List[Any], fallback_text: str) -> dict[str, Any]:
    narrative_chunks: List[str] = []
    evidence_refs: List[Any] = []
    metadata_refs: List[Any] = []
    seen_narrative = set()

    for item in source_refs:
        content = prepare_keypoint_source(get_source_ref_content(item))
        if not content:
            continue
        content_type = get_content_type(item, content)

        if content_type == "metadata":
            metadata_refs.append(item)
            continue
        if content_type == "evidence":
            evidence_refs.append(item)
            continue

        narrative_sentences = extract_narrative_sentences(content)
        narrative_text = " ".join(narrative_sentences).strip() or content
        normalized = re.sub(r"\s+", " ", narrative_text).lower().strip()
        if normalized and normalized not in seen_narrative:
            seen_narrative.add(normalized)
            narrative_chunks.append(narrative_text)

    if not narrative_chunks:
        fallback_narrative = " ".join(extract_narrative_sentences(fallback_text)).strip()
        if fallback_narrative:
            narrative_chunks = pack_semantic_summary_chunks(fallback_narrative)
        else:
            narrative_chunks = pack_semantic_summary_chunks(fallback_text)

    narrative_text = "\n".join(narrative_chunks).strip()
    evidence_lines = build_evidence_reference_lines(evidence_refs + source_refs)

    return {
        "narrative_text": narrative_text,
        "narrative_chunks": narrative_chunks[:MAP_REDUCE_MAX_CHUNKS],
        "evidence_refs": evidence_refs,
        "metadata_refs": metadata_refs,
        "evidence_lines": evidence_lines,
    }


def build_coherent_story_summary(
    narrative_text: str,
    timeline: List[dict],
    fallback_summary: str,
) -> str:
    if timeline:
        events = [
            f"{item.get('marker')}: {item.get('event')}"
            for item in timeline[:4]
            if item.get("event")
        ]
        if events:
            return normalize_formal_statement(" ".join(events))

    if fallback_summary:
        return normalize_formal_statement(fallback_summary)

    keypoints = extract_keypoints(narrative_text, "Narasi Kasus", "ringkasan cerita kasus")
    return normalize_formal_statement(build_overview(keypoints))


def pack_semantic_summary_chunks(text: str, target_chars: int = MAP_REDUCE_CHUNK_TARGET) -> List[str]:
    sentences = split_into_sentences(text)
    if not sentences:
        return [clean_text(text)] if clean_text(text) else []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        projected = current_len + len(sentence) + (1 if current else 0)
        if current and projected > target_chars:
            chunks.append(" ".join(current).strip())
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len = projected

    if current:
        chunks.append(" ".join(current).strip())

    return chunks[:MAP_REDUCE_MAX_CHUNKS]


def build_map_reduce_summary(
    question: str,
    raw_text: str,
    semantic_chunks: List[str] | None = None,
    title: str = "Dokumen Kasus",
) -> tuple[str, List[str], List[dict]]:
    chunks = [chunk for chunk in (semantic_chunks or []) if chunk and chunk.strip()]
    if not chunks:
        chunks = pack_semantic_summary_chunks(raw_text)

    mapped_results: List[dict] = []
    for index, chunk in enumerate(chunks[:MAP_REDUCE_MAX_CHUNKS], start=1):
        keypoints = extract_keypoints(chunk, title, question)
        chunk_summary = build_keyword_summary(question, "", chunk, keypoints)
        chunk_points = build_resume_points(question, chunk, keypoints)[:2]
        if not chunk_summary and not chunk_points:
            continue
        mapped_results.append(
            {
                "chunk_index": index,
                "summary": chunk_summary,
                "points": chunk_points,
            }
        )

    if not mapped_results:
        fallback_keypoints = extract_keypoints(raw_text, title, question)
        fallback_summary = build_keyword_summary(question, title, raw_text, fallback_keypoints)
        fallback_points = build_resume_points(question, raw_text, fallback_keypoints)
        return fallback_summary, fallback_points, []

    reduce_seed_parts: List[str] = []
    for item in mapped_results:
        if item.get("summary"):
            reduce_seed_parts.append(item["summary"])
        reduce_seed_parts.extend(item.get("points") or [])

    reduce_seed = "\n".join(remove_duplicate_sentences(reduce_seed_parts)).strip()
    reduced_keypoints = extract_keypoints(reduce_seed, title, question)
    reduced_points = build_resume_points(question, reduce_seed, reduced_keypoints)
    reduced_summary = build_keyword_summary(question, title, reduce_seed, reduced_points or reduced_keypoints)

    if not reduced_summary:
        reduced_summary = build_overview(reduced_points or reduced_keypoints)

    return reduced_summary, reduced_points, mapped_results


def normalize_actor_name(actor: str) -> str:
    normalized = re.sub(r"\s+", " ", (actor or "")).strip(" .,:;()[]{}")
    normalized = re.sub(r"^[\-:]+", "", normalized).strip()
    normalized = re.sub(r"\b(?:dan|atau|yang|dengan|kepada|terhadap)\b$", "", normalized, flags=re.IGNORECASE).strip()
    return normalized


def is_valid_actor_name(actor: str) -> bool:
    normalized = normalize_actor_name(actor)
    if len(normalized) < 3:
        return False

    lowered = normalized.lower()
    if lowered in ACTOR_GENERIC_TERMS:
        return False

    if normalized in ACTOR_STOPWORDS:
        return False

    parts = lowered.split()
    if not parts:
        return False

    if parts[0] in ACTOR_LEAD_BLACKLIST:
        return False

    if len(parts) == 1 and parts[0] not in ACTOR_ROLE_TERMS and not normalized.isupper():
        return False

    if all(part in ACTOR_GENERIC_TERMS for part in parts):
        return False

    return True


def score_actor_candidate(actor: str, sentence: str) -> float:
    normalized = normalize_actor_name(actor)
    parts = normalized.split()
    score = len(parts) * 2

    if any(term == part.lower() for part in parts for term in ACTOR_ROLE_TERMS):
        score += 4
    if normalized.isupper():
        score += 1.5
    if len(parts) >= 2:
        score += 2
    if re.search(r"\b(?:PT|CV|Bank|Direktur|Manajer|Saksi|Terdakwa|Korban)\b", normalized, re.IGNORECASE):
        score += 3
    if re.search(r"\b(?:kepada|oleh|dengan|terhadap)\b", sentence, re.IGNORECASE):
        score += 0.5

    return score


def dedupe_actor_candidates(candidates: List[tuple[str, float]]) -> List[str]:
    ordered = sorted(candidates, key=lambda item: (item[1], len(item[0])), reverse=True)
    selected: List[str] = []

    for actor, _ in ordered:
        normalized = actor.lower()
        if any(
            normalized == existing.lower()
            or normalized in existing.lower()
            or existing.lower() in normalized
            for existing in selected
        ):
            continue
        selected.append(actor)
        if len(selected) >= 16:
            break

    return selected


def classify_actor_role(actor: str, source_text: str) -> str:
    normalized_actor = normalize_actor_name(actor)
    actor_pattern = re.escape(normalized_actor)
    context_matches = re.findall(
        rf"([^.:\n]{{0,80}}{actor_pattern}[^.:\n]{{0,80}})",
        source_text,
        flags=re.IGNORECASE,
    )
    context_text = " ".join(context_matches) if context_matches else normalized_actor

    for role, patterns in FORENSIC_ROLE_PATTERNS:
        if any(re.search(pattern, normalized_actor, re.IGNORECASE) for pattern in patterns):
            return role
        if any(re.search(pattern, context_text, re.IGNORECASE) for pattern in patterns):
            return role

    return "Pihak Lain"


def build_actor_profiles(actors: List[str], source_text: str) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for actor in actors:
        profiles[actor] = {
            "name": actor,
            "role": classify_actor_role(actor, source_text),
            "context": extract_actor_context(actor, source_text),
        }
    return profiles


def extract_actor_context(actor: str, source_text: str) -> str:
    if not actor or not source_text:
        return ""

    actor_pattern = re.escape(normalize_actor_name(actor))
    candidates = [
        sentence for sentence in split_into_sentences(source_text)
        if re.search(actor_pattern, sentence, re.IGNORECASE)
    ]
    if not candidates:
        return ""

    candidates.sort(key=lambda item: (sentence_information_score(item), len(item)), reverse=True)
    return summarize_keypoint(candidates[0], max_words=42)


def extract_actor_contexts(actor: str, source_text: str, limit: int = 2) -> List[str]:
    if not actor or not source_text:
        return []

    actor_pattern = re.escape(normalize_actor_name(actor))
    candidates = [
        sentence for sentence in split_into_sentences(source_text)
        if re.search(actor_pattern, sentence, re.IGNORECASE)
    ]
    if not candidates:
        return []

    candidates.sort(key=lambda item: (sentence_information_score(item), len(item)), reverse=True)
    selected: List[str] = []
    seen = set()
    for candidate in candidates:
        summary = summarize_keypoint(candidate, max_words=44)
        key = summary.lower()
        if not summary or key in seen:
            continue
        seen.add(key)
        selected.append(summary)
        if len(selected) >= limit:
            break
    return selected


def build_actor_role_lines(actor_profiles: dict[str, dict], actors: List[str]) -> List[str]:
    lines: List[str] = []
    for actor in actors:
        role = actor_profiles.get(actor, {}).get("role", "Pihak Lain")
        context = (actor_profiles.get(actor, {}).get("context") or "").strip()
        if context:
            lines.append(f"- {actor} [{role}]: {context}")
        else:
            lines.append(f"- {actor} [{role}]")
    return lines


def build_actor_case_lines(
    actor_profiles: dict[str, dict],
    actors: List[str],
    source_text: str,
) -> List[str]:
    lines: List[str] = []
    for actor in actors:
        role = actor_profiles.get(actor, {}).get("role", "Pihak Lain")
        contexts = extract_actor_contexts(actor, source_text, limit=2)
        if contexts:
            detail = " ".join(normalize_formal_statement(item) for item in contexts)
            lines.append(f"- {actor} [{role}]: {detail}")
        else:
            lines.append(f"- {actor} [{role}]")
    return lines


def find_timeline_marker_for_sentence(sentence: str) -> str:
    marker_match = TIMELINE_MARKER_REGEX.search(sentence or "")
    if marker_match:
        return marker_match.group(0)

    lowered = (sentence or "").lower()
    relative_markers = (
        "awalnya", "sebelumnya", "hari yang sama", "kemudian",
        "selanjutnya", "setelah itu", "hari berikutnya",
        "minggu berikutnya", "bulan berikutnya", "akhirnya",
    )
    for marker in relative_markers:
        if marker in lowered:
            return marker
    return ""


def build_investigative_points(
    source_text: str,
    actors: List[str],
    relationships: List[tuple[str, str, str, str]],
) -> List[str]:
    if not source_text:
        return []

    actor_names = actors or extract_actor_candidates(source_text)
    actor_lookup = {actor.lower(): actor for actor in actor_names}
    relationship_lookup = {
        (left.lower(), relation.lower(), right.lower()): evidence
        for left, relation, right, evidence in relationships
    }
    points: List[tuple[float, str]] = []
    seen = set()

    for sentence in split_into_sentences(source_text):
        lowered = sentence.lower()
        present_actors = [
            actor for actor in actor_names
            if actor.lower() in lowered
        ]

        action = detect_relation_label(sentence)
        if len(present_actors) >= 2:
            pair = select_relationship_pair(sentence, present_actors, action)
        else:
            pair = None

        actor_from = pair[0] if pair else (present_actors[0] if present_actors else "")
        actor_to = pair[1] if pair else (present_actors[1] if len(present_actors) > 1 else "")
        when = find_timeline_marker_for_sentence(sentence)

        if not actor_from and not action and not when:
            continue

        parts = []
        if actor_from:
            parts.append(actor_lookup.get(actor_from.lower(), actor_from))
        if action and action != "terkait":
            parts.append(action)
        elif actor_from:
            parts.append("terlibat")
        if actor_to:
            parts.append(f"kepada {actor_lookup.get(actor_to.lower(), actor_to)}")
        if when:
            parts.append(f"pada {when}")

        if len(parts) < 2:
            continue

        point = " - ".join(parts)
        evidence = relationship_lookup.get(
            (actor_from.lower(), action.lower(), actor_to.lower()),
            sentence,
        ) if actor_from and actor_to else sentence
        detail = summarize_keypoint(evidence, max_words=28)
        if detail and detail.lower() not in point.lower():
            point = f"{point}: {detail}"

        key = point.lower()
        if key in seen:
            continue
        seen.add(key)
        score = sentence_information_score(sentence, extract_query_terms(""))
        score += 4 if when else 0
        score += 4 if actor_from and actor_to else 0
        score += 2 if action and action != "terkait" else 0
        points.append((score, point))

    points.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
    return [point for _, point in points[:5]]


def build_relationship_lines_detailed(
    relationships: List[tuple[str, str, str, str]],
    actor_profiles: dict[str, dict],
) -> List[str]:
    lines: List[str] = []
    for left, relation, right, evidence in relationships[:6]:
        left_role = actor_profiles.get(left, {}).get("role", "Pihak Lain")
        right_role = actor_profiles.get(right, {}).get("role", "Pihak Lain")
        when = find_timeline_marker_for_sentence(evidence)
        evidence_detail = summarize_keypoint(evidence, max_words=42)
        parts = [f"{left} [{left_role}] {relation} {right} [{right_role}]"]
        if when:
            parts.append(f"terjadi pada {when}")
        statement = join_formal_clauses(parts)
        if evidence_detail and evidence_detail.lower() not in statement.lower():
            statement = normalize_formal_statement(f"{statement.rstrip('.')} Bukti atau konteks yang teridentifikasi: {evidence_detail}")
        lines.append(f"- {statement}")
    return lines


def build_timeline_lines_detailed(
    timeline: List[dict],
    actor_profiles: dict[str, dict],
) -> List[str]:
    lines: List[str] = []
    for item in timeline[:5]:
        marker = item.get("marker") or "Tahap"
        event = item.get("event") or ""
        participants = item.get("participants") or []
        if participants:
            participant_labels = [
                f"{actor} [{actor_profiles.get(actor, {}).get('role', 'Pihak Lain')}]"
                for actor in participants[:4]
            ]
            statement = normalize_formal_statement(f"{marker}: {event}")
            lines.append(f"- {statement} Pihak yang terlibat: {', '.join(participant_labels)}.")
        else:
            lines.append(f"- {normalize_formal_statement(f'{marker}: {event}')}")
    return lines


def order_actors_by_role(actor_profiles: dict[str, dict], actors: List[str]) -> List[str]:
    role_rank = {role: index for index, role in enumerate(ROLE_DISPLAY_ORDER)}
    return sorted(
        actors,
        key=lambda actor: (
            role_rank.get(actor_profiles.get(actor, {}).get("role", "Pihak Lain"), 999),
            actor_profiles.get(actor, {}).get("name", actor).lower(),
        ),
    )


def select_case_diagram_actors(
    actor_profiles: dict[str, dict],
    actors: List[str],
    relationships: List[tuple[str, str, str, str]],
) -> List[str]:
    ordered_actors = order_actors_by_role(actor_profiles, actors)
    core_actors = [
        actor for actor in ordered_actors
        if actor_profiles.get(actor, {}).get("role") in CASE_CORE_ROLES
    ]

    if core_actors:
        selected = core_actors[:12]
        selected_set = {actor.lower() for actor in selected}

        # Keep non-core actors only when they bridge a relationship between core actors
        # and the diagram would otherwise lose an important edge.
        for left, _, right, _ in relationships:
            left_role = actor_profiles.get(left, {}).get("role", "Pihak Lain")
            right_role = actor_profiles.get(right, {}).get("role", "Pihak Lain")
            if left_role in CASE_SUPPORT_ROLES and right.lower() in selected_set:
                continue
            if right_role in CASE_SUPPORT_ROLES and left.lower() in selected_set:
                continue

        return selected

    # Fallback when the document mostly contains legal process actors.
    return ordered_actors[:12]


def filter_relationships_for_actors(
    relationships: List[tuple[str, str, str, str]],
    allowed_actors: List[str],
) -> List[tuple[str, str, str, str]]:
    allowed = {actor.lower() for actor in allowed_actors}
    filtered = [
        item for item in relationships
        if item[0].lower() in allowed and item[2].lower() in allowed
    ]
    return filtered[:18]


def extract_actor_candidates(text: str) -> List[str]:
    sentences = split_into_sentences(text)
    scored_candidates: List[tuple[str, float]] = []
    patterns = [
        re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,3})\b"),
        re.compile(
            r"(?i)\b(?:korban|pelaku|tersangka|terdakwa|saksi|penyidik|jaksa|hakim|pengacara|kuasa hukum|"
            r"perusahaan|bank|instansi|direktur|manajer|pegawai|karyawan|nasabah|klien)"
            r"(?:\s+[A-Z][a-z]+){0,3}\b"
        ),
    ]

    for sentence in sentences:
        for pattern in patterns:
            for match in pattern.findall(sentence):
                actor = normalize_actor_name(match)
                if not is_valid_actor_name(actor):
                    continue
                scored_candidates.append((actor, score_actor_candidate(actor, sentence)))

    return dedupe_actor_candidates(scored_candidates)


def detect_relation_label(sentence: str) -> str:
    for pattern, label in RELATION_PATTERNS:
        if re.search(pattern, sentence):
            return label
    return "terkait"


def find_relation_pattern(sentence: str) -> str:
    for pattern, _ in RELATION_PATTERNS:
        if re.search(pattern, sentence):
            return pattern
    return ""


def select_relationship_pair(
    sentence: str,
    present_actors: List[str],
    relation: str,
) -> tuple[str, str] | None:
    if len(present_actors) < 2:
        return None

    if relation == "terkait":
        return present_actors[0], present_actors[1]

    relation_pattern = find_relation_pattern(sentence)
    relation_match = re.search(relation_pattern, sentence) if relation_pattern else None
    relation_index = relation_match.start() if relation_match else len(sentence) // 2

    actor_positions = []
    lowered_sentence = sentence.lower()
    for actor in present_actors:
        position = lowered_sentence.find(actor.lower())
        if position >= 0:
            actor_positions.append((position, actor))

    if len(actor_positions) < 2:
        return present_actors[0], present_actors[1]

    actor_positions.sort(key=lambda item: item[0])
    left_candidates = [item for item in actor_positions if item[0] <= relation_index]
    right_candidates = [item for item in actor_positions if item[0] > relation_index]

    if left_candidates and right_candidates:
        left = left_candidates[-1][1]
        right = right_candidates[0][1]
        if left != right:
            return left, right

    return actor_positions[0][1], actor_positions[1][1]


def score_relationship_sentence(sentence: str, relation: str, actor_count: int) -> float:
    score = actor_count * 3
    score += 4 if relation != "terkait" else 0
    score += min(len(sentence.split()), 24) / 8
    if any(token in sentence.lower() for token in ("bukti", "transfer", "laporan", "perintah", "komunikasi", "rekening")):
        score += 2
    return score


def build_timeline_sort_key(marker: str, index: int) -> tuple[int, int, int, int]:
    normalized = (marker or "").strip().lower()
    month_map = {
        "january": 1, "januari": 1,
        "february": 2, "februari": 2,
        "march": 3, "maret": 3,
        "april": 4,
        "may": 5, "mei": 5,
        "june": 6, "juni": 6,
        "july": 7, "juli": 7,
        "august": 8, "agustus": 8,
        "september": 9,
        "october": 10, "oktober": 10,
        "november": 11,
        "december": 12, "desember": 12,
    }

    match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", normalized)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3)), index

    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", normalized)
    if match:
        year = int(match.group(3))
        if year < 100:
            year += 2000
        return year, int(match.group(2)), int(match.group(1)), index

    match = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{2,4})", normalized)
    if match:
        year = int(match.group(3))
        if year < 100:
            year += 2000
        return year, month_map.get(match.group(2), 0), int(match.group(1)), index

    match = re.search(r"([a-z]+)\s+(\d{4})", normalized)
    if match:
        return int(match.group(2)), month_map.get(match.group(1), 0), 0, index

    relative_rank = {
        "awalnya": 0,
        "sebelumnya": 1,
        "hari yang sama": 2,
        "kemudian": 3,
        "selanjutnya": 4,
        "setelah itu": 5,
        "hari berikutnya": 6,
        "minggu berikutnya": 7,
        "bulan berikutnya": 8,
        "akhirnya": 9,
    }
    for phrase, rank in relative_rank.items():
        if phrase in normalized:
            return 9999, 0, rank, index

    return 9999, 99, 99, index


def score_timeline_sentence(sentence: str, marker: str, actors: List[str]) -> float:
    lowered = sentence.lower()
    score = 4 if marker else 0
    score += min(len(sentence.split()), 28) / 10
    score += sum(1 for keyword in TIMELINE_EVENT_KEYWORDS if keyword in lowered) * 1.2
    score += sum(1 for actor in actors if actor.lower() in lowered) * 1.5
    if any(token in lowered for token in ("laporan", "transfer", "rekening", "pertemuan", "investigasi", "bukti", "pemeriksaan")):
        score += 2
    return score


def extract_forensic_timeline(text: str, actors: List[str] | None = None) -> List[dict]:
    sentences = split_into_sentences(text)
    actor_names = actors or extract_actor_candidates(text)
    timeline_candidates: List[tuple[tuple[int, int, int, int], float, dict]] = []
    seen = set()

    for index, sentence in enumerate(sentences):
        lowered = sentence.lower()
        marker_match = TIMELINE_MARKER_REGEX.search(sentence)
        marker = marker_match.group(0) if marker_match else ""
        has_sequence_keyword = any(keyword in lowered for keyword in TIMELINE_EVENT_KEYWORDS)

        if not marker and not has_sequence_keyword:
            continue

        event = summarize_keypoint(sentence, max_words=26)
        if not is_useful_keypoint(event):
            continue

        participants = [
            actor for actor in actor_names
            if actor.lower() in lowered
        ][:3]
        sort_key = build_timeline_sort_key(marker or lowered, index)
        score = score_timeline_sentence(sentence, marker, actor_names)
        dedupe_key = (marker.lower(), event.lower()) if marker else event.lower()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        timeline_candidates.append((
            sort_key,
            score,
            {
                "marker": marker or f"Tahap {index + 1}",
                "event": event,
                "participants": participants,
            },
        ))

    timeline_candidates.sort(key=lambda item: (item[0], -item[1]))
    return [payload for _, _, payload in timeline_candidates[:8]]


def build_social_relationships(text: str) -> tuple[List[str], List[tuple[str, str, str, str]]]:
    actors = extract_actor_candidates(text)
    if not actors:
        return [], []

    actor_lookup = {actor.lower(): actor for actor in actors}
    actor_names = list(actor_lookup.values())
    sentences = split_into_sentences(text)
    relationship_candidates: List[tuple[float, str, str, str, str]] = []
    seen_edges = set()

    for sentence in sentences:
        present = []
        lowered_sentence = sentence.lower()
        for actor in actor_names:
            if actor.lower() in lowered_sentence:
                present.append(actor)

        if len(present) < 2:
            continue

        relation = detect_relation_label(sentence)
        pair = select_relationship_pair(sentence, present, relation)
        if not pair:
            continue
        left, right = pair
        if left == right:
            continue
        edge_key = (left.lower(), relation, right.lower())
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        evidence = summarize_keypoint(sentence, max_words=22)
        score = score_relationship_sentence(sentence, relation, len(present))
        relationship_candidates.append((score, left, relation, right, evidence))

    relationship_candidates.sort(key=lambda item: (item[0], len(item[4])), reverse=True)
    relationships = [
        (left, relation, right, evidence)
        for _, left, relation, right, evidence in relationship_candidates[:18]
    ]

    used_actors = []
    seen_actors = set()
    for left, _, right, _ in relationships:
        if left.lower() not in seen_actors:
            seen_actors.add(left.lower())
            used_actors.append(left)
        if right.lower() not in seen_actors:
            seen_actors.add(right.lower())
            used_actors.append(right)

    if used_actors:
        return used_actors[:14], relationships

    return actor_names[:14], relationships


def build_mermaid_graph(
    actors: List[str],
    relationships: List[tuple[str, str, str, str]],
    actor_profiles: dict[str, dict],
) -> str:
    if not actors:
        return ""

    ordered_actors = order_actors_by_role(actor_profiles, actors)
    node_ids = {
        actor: f"A{index}"
        for index, actor in enumerate(ordered_actors, start=1)
    }
    lines = ["```mermaid", "flowchart LR", '    CASE["Kasus"]']

    role_groups: dict[str, List[str]] = {}
    for actor in ordered_actors:
        role = actor_profiles.get(actor, {}).get("role", "Pihak Lain")
        role_groups.setdefault(role, []).append(actor)

    for role in ROLE_DISPLAY_ORDER:
        members = role_groups.get(role, [])
        if not members:
            continue
        safe_role = role.replace('"', "'")
        subgroup_id = re.sub(r"[^A-Za-z0-9]", "", role.upper()) or "GROUP"
        lines.append(f'    subgraph {subgroup_id}["{safe_role}"]')
        for actor in members:
            node_id = node_ids[actor]
            safe_label = actor.replace('"', "'")
            lines.append(f'        {node_id}["{safe_label}"]')
        lines.append("    end")

    if relationships:
        connected = set()
        for left, relation, right, _ in relationships:
            if left not in node_ids or right not in node_ids:
                continue
            safe_relation = relation.replace('"', "'")
            lines.append(
                f'    {node_ids[left]} -- "{safe_relation}" --> {node_ids[right]}'
            )
            connected.add(left)
            connected.add(right)
        for actor in actors:
            if actor not in connected:
                lines.append(f'    CASE -. "terkait" .-> {node_ids[actor]}')
    else:
        for actor in actors[:6]:
            lines.append(f'    CASE -- "terkait" --> {node_ids[actor]}')

    lines.append("    classDef case fill:#f3f0ff,stroke:#4c1d95,stroke-width:2px,color:#1f1f1f;")
    lines.append("    classDef korban fill:#fee2e2,stroke:#b91c1c,color:#1f1f1f;")
    lines.append("    classDef pelaku fill:#fde68a,stroke:#b45309,color:#1f1f1f;")
    lines.append("    classDef saksi fill:#dbeafe,stroke:#1d4ed8,color:#1f1f1f;")
    lines.append("    classDef penyidik fill:#dcfce7,stroke:#15803d,color:#1f1f1f;")
    lines.append("    classDef hukum fill:#fae8ff,stroke:#a21caf,color:#1f1f1f;")
    lines.append("    classDef instansi fill:#e0f2fe,stroke:#0369a1,color:#1f1f1f;")
    lines.append("    classDef lain fill:#f3f4f6,stroke:#4b5563,color:#1f1f1f;")
    lines.append("    class CASE case;")
    for actor in ordered_actors:
        role = actor_profiles.get(actor, {}).get("role", "Pihak Lain")
        class_name = {
            "Korban": "korban",
            "Pelaku": "pelaku",
            "Saksi": "saksi",
            "Penyidik": "penyidik",
            "Penegak Hukum": "hukum",
            "Kuasa Hukum": "hukum",
            "Instansi": "instansi",
            "Pihak Lain": "lain",
        }.get(role, "lain")
        lines.append(f"    class {node_ids[actor]} {class_name};")
    lines.append("```")
    return "\n".join(lines)


def build_timeline_mermaid(timeline: List[dict]) -> str:
    if not timeline:
        return ""

    lines = ["```mermaid", "flowchart TB"]

    for index, item in enumerate(timeline[:6], start=1):
        marker = (item.get("marker") or f"Tahap {index}").replace('"', "'")
        event = (item.get("event") or "").replace('"', "'")
        participants = item.get("participants") or []
        participant_text = ""
        if participants:
            participant_text = "\\nPihak: " + ", ".join(participants[:3]).replace('"', "'")
        label = f"{marker}\\n{event}{participant_text}"
        lines.append(f'    T{index}["{label}"]')

    for index in range(1, min(len(timeline), 6)):
        lines.append(f'    T{index} --> T{index + 1}')

    lines.append("    classDef timeline fill:#fff7ed,stroke:#c2410c,stroke-width:2px,color:#1f1f1f;")
    for index in range(1, min(len(timeline), 6) + 1):
        lines.append(f"    class T{index} timeline;")
    lines.append("```")
    return "\n".join(lines)


def sanitize_mermaid_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", (text or "")).strip()
    normalized = normalized.replace('"', "'")
    normalized = normalized.replace("[", "(").replace("]", ")")
    normalized = normalized.replace("{", "(").replace("}", ")")
    normalized = normalized.replace("#", "")
    normalized = normalized.replace(";", ",")
    return normalized


def truncate_mermaid_text(text: str, max_words: int = 14) -> str:
    normalized = sanitize_mermaid_text(text)
    words = normalized.split()
    if len(words) <= max_words:
        return normalized
    return " ".join(words[:max_words]).rstrip(",.:") + "..."


def categorize_relation_edge(relation: str) -> str:
    normalized = (relation or "").strip().lower()
    if normalized in FLOW_RELATION_LABELS:
        return "flow"
    if normalized in COMMUNICATION_RELATION_LABELS:
        return "communication"
    if normalized in LEGAL_RELATION_LABELS:
        return "legal"
    return "neutral"


def determine_edge_strength(relation: str, evidence: str) -> str:
    normalized_relation = (relation or "").strip().lower()
    normalized_evidence = (evidence or "").strip().lower()

    if any(term in normalized_evidence for term in STRONG_EVIDENCE_TERMS):
        return "strong"

    if any(term in normalized_evidence for term in WEAK_EVIDENCE_TERMS):
        return "weak"

    if normalized_relation in FLOW_RELATION_LABELS | LEGAL_RELATION_LABELS:
        return "strong"

    if normalized_relation in COMMUNICATION_RELATION_LABELS:
        return "medium"

    return "weak"


def build_edge_style(category: str, strength: str) -> str:
    palette = {
        "flow": "#0f766e",
        "communication": "#2563eb",
        "legal": "#7c3aed",
        "neutral": "#6b7280",
        "timeline": "#c2410c",
        "timeline-link": "#ea580c",
        "case-link": "#9ca3af",
    }
    color = palette.get(category, palette["neutral"])

    if strength == "strong":
        return f"stroke:{color},stroke-width:4px;"
    if strength == "medium":
        return f"stroke:{color},stroke-width:2.5px;"
    return f"stroke:{color},stroke-width:1.75px,stroke-dasharray: 6 4;"


def build_integrated_case_mermaid(
    actors: List[str],
    relationships: List[tuple[str, str, str, str]],
    actor_profiles: dict[str, dict],
    timeline: List[dict],
    central_nodes: List[str] | None = None,
) -> str:
    if not actors:
        return ""

    ordered_actors = order_actors_by_role(actor_profiles, actors) if actors else []
    central_node_set = {node for node in (central_nodes or []) if node}
    node_ids = {
        actor: f"A{index}"
        for index, actor in enumerate(ordered_actors, start=1)
    }
    lines = ["```mermaid", "flowchart LR", '    CASE["Kasus Investigasi"]']
    link_styles: List[str] = []

    for actor in ordered_actors:
        profile = actor_profiles.get(actor, {})
        role = profile.get("role", "Pihak Lain")
        context = truncate_mermaid_text(profile.get("context", ""), max_words=12)
        safe_name = truncate_mermaid_text(actor, max_words=8)
        safe_role = sanitize_mermaid_text(role)
        safe_context = f"\\n{context}" if context else ""
        safe_label = f"{safe_name}\\n({safe_role}){safe_context}"
        lines.append(f'    {node_ids[actor]}["{safe_label}"]')

    if ordered_actors:
        connected = set()
        if relationships:
            for left, relation, right, evidence in relationships:
                if left not in node_ids or right not in node_ids:
                    continue
                evidence_label = truncate_mermaid_text(evidence, max_words=11)
                safe_relation = sanitize_mermaid_text(relation)
                edge_label = safe_relation
                if evidence_label:
                    edge_label = f"{safe_relation}\\n{evidence_label}"
                category = categorize_relation_edge(relation)
                strength = determine_edge_strength(relation, evidence)
                if strength == "weak":
                    lines.append(f'    {node_ids[left]} -. "{edge_label}" .-> {node_ids[right]}')
                else:
                    lines.append(f'    {node_ids[left]} -- "{edge_label}" --> {node_ids[right]}')
                link_styles.append(build_edge_style(category, strength))
                connected.add(left)
                connected.add(right)
        for actor in ordered_actors:
            if actor not in connected:
                lines.append(f'    CASE -. "terkait" .-> {node_ids[actor]}')
                link_styles.append(build_edge_style("case-link", "weak"))

    lines.append("    classDef case fill:#f8fafc,stroke:#334155,stroke-width:2px,color:#0f172a;")
    lines.append("    classDef actor fill:#eef2ff,stroke:#4338ca,stroke-width:2px,color:#111827;")
    lines.append("    classDef central fill:#fff7ed,stroke:#c2410c,stroke-width:3px,color:#111827;")
    lines.append("    class CASE case;")
    for actor in ordered_actors:
        lines.append(f"    class {node_ids[actor]} actor;")
        if actor in central_node_set:
            lines.append(f"    class {node_ids[actor]} central;")
    for index, style in enumerate(link_styles):
        lines.append(f"    linkStyle {index} {style}")
    lines.append("```")
    return "\n".join(lines)


def format_case_summary(
    question: str,
    raw_text: str,
    semantic_chunks: List[str] | None = None,
) -> tuple[str, List[str], str]:
    title = "Dokumen Kasus"
    summary, resume_points, mapped_results = build_map_reduce_summary(
        question,
        raw_text,
        semantic_chunks,
        title=title,
    )

    output = ["Executive Summary", ""]
    if summary:
        output.append(f"- {normalize_formal_statement(summary)}")
        output.append("")

    output.append("Ringkasan Kasus")
    for point in resume_points[:5]:
        output.append(f"- {point}")

    if mapped_results:
        output.append("")
        output.append("Jejak Map-Reduce")
        for item in mapped_results[:4]:
            chunk_label = f"Bagian {item.get('chunk_index', 0)}"
            chunk_summary = (item.get("summary") or "").strip()
            if chunk_summary:
                output.append(f"- {chunk_label}: {summarize_keypoint(chunk_summary, max_words=26)}")

    return "\n".join(output).strip(), resume_points, summary


def format_case_analysis(
    workspace_id: str,
    question: str,
    raw_text: str,
    semantic_chunks: List[str] | None = None,
    source_refs: List | None = None,
) -> tuple[str, List[str], str]:
    title = "Dokumen Kasus"
    keypoints = extract_keypoints(raw_text, title, question)
    summary, resume_points, mapped_results = build_map_reduce_summary(
        question,
        raw_text,
        semantic_chunks,
        title=title,
    )
    actors, relationships = build_social_relationships(raw_text)
    actor_profiles = build_actor_profiles(actors, raw_text)
    graph_bundle = build_graph_bundle(
        workspace_id=workspace_id,
        question=question,
        semantic_chunks=semantic_chunks or pack_semantic_summary_chunks(raw_text),
        actors=actors,
        relationships=relationships,
        actor_profiles=actor_profiles,
    )
    petitioner_identity = extract_petitioner_identity(source_refs or [], raw_text)
    investigative_points = build_investigative_points(raw_text, actors, relationships)
    issue_terms = (
        "dugaan", "kerugian", "pelanggaran", "konflik", "bukti",
        "fraud", "sengketa", "transfer", "laporan", "investigasi",
    )
    issue_lines = [
        summarize_keypoint(sentence, max_words=38)
        for sentence in split_into_sentences(raw_text)
        if any(term in sentence.lower() for term in issue_terms)
    ]
    issue_lines = remove_duplicate_sentences(issue_lines)[:4]

    output = ["Laporan Analisis Kasus", ""]
    if petitioner_identity:
        output.append("Identitas Pemohon Kasasi")
        for label in IDENTITY_FIELDS:
            value = petitioner_identity.get(label)
            if value:
                output.append(f"- {label}: {value}")
        output.append("")

    if summary:
        output.append("Ringkasan Eksekutif")
        output.append(f"- {summary}")
        output.append("")

    if actors:
        output.append("Identifikasi Pihak")
        output.extend(build_actor_case_lines(actor_profiles, actors[:6], raw_text))
        output.append("")

    if graph_bundle.central_nodes:
        output.append("Aktor Sentral")
        for actor in graph_bundle.central_nodes[:4]:
            role = actor_profiles.get(actor, {}).get("role", "Pihak Lain")
            output.append(f"- {actor} [{role}]")
        output.append("")

    output.append("Temuan Utama")
    if investigative_points:
        for point in investigative_points[:5]:
            output.append(f"- {point}")
    else:
        for point in (resume_points[:4] or keypoints[:4] or ["Tidak ada temuan utama yang cukup jelas dari dokumen."]):
            output.append(f"- {summarize_keypoint(point, max_words=40)}")
    output.append("")

    output.append("Analisis Relasi")
    if relationships:
        output.extend(build_relationship_lines_detailed(relationships[:4], actor_profiles))
    else:
        output.append("- Relasi antar pihak belum tergambar secara eksplisit pada teks yang tersedia.")

    if graph_bundle.local_search_lines:
        output.append("")
        output.append("GraphRAG Local Search")
        output.extend(graph_bundle.local_search_lines[:5])

    if graph_bundle.cypher_path_lines:
        output.append("")
        output.append("Jalur Relasi Cypher")
        output.extend(graph_bundle.cypher_path_lines[:5])

    if graph_bundle.communities:
        output.append("")
        output.append("GraphRAG Community View")
        for community in graph_bundle.communities[:3]:
            output.append(f"- {community.get('summary', '')}")

    if issue_lines:
        output.append("")
        output.append("Catatan Investigatif")
        for item in issue_lines:
            output.append(f"- {item}")

    if mapped_results:
        output.append("")
        output.append("Jejak Map-Reduce")
        for item in mapped_results[:4]:
            chunk_summary = (item.get("summary") or "").strip()
            if chunk_summary:
                output.append(f"- Bagian {item.get('chunk_index', 0)}: {summarize_keypoint(chunk_summary, max_words=24)}")

    return "\n".join(output).strip(), keypoints[:4], summary


def format_social_graph(
    workspace_id: str,
    question: str,
    raw_text: str,
    semantic_chunks: List[str] | None = None,
    source_refs: List | None = None,
) -> tuple[str, List[str], str]:
    title = "Dokumen Kasus"
    analysis = analyze_documents_sync(source_refs or [], raw_text)
    narrative_text = "\n".join(chunk.text for chunk in analysis.narrative_chunks).strip() or raw_text
    keypoints = extract_keypoints(narrative_text, title, question)
    event_actors = top_actors(analysis.events, limit=14)
    graph_relationships_from_story = analysis.story_graph.actor_relations
    if graph_relationships_from_story:
        actors = event_actors or list({
            actor
            for left, _, right, _ in graph_relationships_from_story
            for actor in (left, right)
            if actor
        })
        relationships = graph_relationships_from_story
    else:
        actors, relationships = build_social_relationships(narrative_text)
    actor_profiles = build_actor_profiles(actors, narrative_text)
    graph_bundle = build_graph_bundle(
        workspace_id=workspace_id,
        question=question,
        semantic_chunks=[chunk.text for chunk in analysis.narrative_chunks] or semantic_chunks or pack_semantic_summary_chunks(narrative_text),
        actors=actors,
        relationships=relationships,
        actor_profiles=actor_profiles,
    )
    graph_actor_profiles = graph_bundle_to_actor_profiles(graph_bundle, actor_profiles)
    graph_relationships = graph_bundle_to_relationships(graph_bundle) or relationships
    graph_actors = [node.get("name", "") for node in graph_bundle.nodes if node.get("name")] or actors
    case_actors = select_case_diagram_actors(graph_actor_profiles, graph_actors, graph_relationships)
    case_relationships = filter_relationships_for_actors(graph_relationships, case_actors)
    ordered_actors = order_actors_by_role(graph_actor_profiles, case_actors)
    integrated_mermaid = build_integrated_case_mermaid(
        ordered_actors,
        case_relationships,
        graph_actor_profiles,
        [],
        graph_bundle.central_nodes,
    )

    if integrated_mermaid:
        narrative_summary = analysis.narrative_summary or build_keyword_summary(
            "ringkas narasi utama kasus sebelum diagram jaringan sosial",
            title,
            narrative_text,
            keypoints,
        )
        citation_pages = sorted({
            event.page + 1
            for event in analysis.events
            if event.page is not None
        })[:8]
        citation_text = f" [hal. {', '.join(str(page) for page in citation_pages)}]" if citation_pages else ""
        output = ["Narasi Utama", ""]
        if narrative_summary:
            output.append(f"- {normalize_formal_statement(narrative_summary)}{citation_text}")
        else:
            output.append("- Narasi utama belum cukup jelas dari teks PDF yang diekstrak.")
        output.append("")
        output.append(integrated_mermaid)
        return "\n".join(output).strip(), keypoints[:4], narrative_summary

    return "Diagram investigasi belum dapat dibuat karena aktor atau relasi belum cukup jelas dari narasi PDF.", keypoints[:4], ""

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


def format_graph_path_answer(question: str, graph_lines: List[str]) -> tuple[str, List[str], str]:
    output = ["Analisis Jalur Relasi", ""]
    output.append(f"- Pertanyaan: {normalize_formal_statement(question).rstrip('.')}")
    output.append("")
    output.append("Jalur Keterhubungan")
    output.extend(graph_lines[:6] or ["- Jalur relasi belum dapat ditentukan dari graph yang tersedia."])
    summary = summarize_keypoint(graph_lines[0], max_words=32) if graph_lines else ""
    return "\n".join(output).strip(), graph_lines[:4], summary


def graph_bundle_to_relationships(graph_bundle) -> List[tuple[str, str, str, str]]:
    relationships: List[tuple[str, str, str, str]] = []
    for edge in graph_bundle.edges or []:
        left = edge.get("left", "")
        relation = edge.get("relation", "terkait")
        right = edge.get("right", "")
        evidence = edge.get("evidence", "")
        if left and right:
            relationships.append((left, relation, right, evidence))
    return relationships


def graph_bundle_to_actor_profiles(graph_bundle, fallback_profiles: dict[str, dict]) -> dict[str, dict]:
    profiles = dict(fallback_profiles)
    for node in graph_bundle.nodes or []:
        name = node.get("name", "")
        if not name:
            continue
        profiles[name] = {
            "name": name,
            "role": node.get("role", fallback_profiles.get(name, {}).get("role", "Pihak Lain")),
            "context": node.get("context", fallback_profiles.get(name, {}).get("context", "")),
        }
    return profiles


def build_graphrag_status_lines(graph_bundle) -> List[str]:
    status_lines: List[str] = []
    capabilities = graph_bundle.capabilities or {}
    enabled_components = []
    if capabilities.get("networkx"):
        enabled_components.append("NetworkX graph engine")
    if graph_bundle.neo4j_synced:
        enabled_components.append("Neo4j sync")
    if graph_bundle.cypher_path_lines:
        enabled_components.append("Cypher path query")
    if graph_bundle.kg_triplets:
        enabled_components.append("LlamaIndex KG triplets")
    if capabilities.get("microsoft_graphrag"):
        enabled_components.append("Microsoft GraphRAG package")

    if enabled_components:
        status_lines.append(f"- Pipeline GraphRAG aktif melalui: {', '.join(enabled_components)}.")
    else:
        status_lines.append("- Pipeline graph aktif dengan mode fallback lokal.")

    if graph_bundle.central_nodes:
        status_lines.append(
            f"- Aktor sentral yang diprioritaskan: {', '.join(graph_bundle.central_nodes[:4])}."
        )

    return status_lines


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

        mode = detect_response_mode(question)
        if mode in DISABLED_RESPONSE_MODES:
            return {
                "answer": DISABLED_MODE_ANSWER,
                "highlight": [],
                "highlight_pages": [],
                "sources": [],
                "selected_section_title": "",
                "summary": "",
            }

        best, selected_title, selected_text = select_best_doc_match(docs, question)
        if best is None:
            return {
                "answer": NO_ANSWER,
                "sources": []
            }

        page = best.metadata.get("page", 0)

        if mode in {"summary", "analysis", "social_graph"}:
            combined_text, source_refs = collect_document_pages(
                workspace_id,
                best.metadata.get("source", ""),
                docs,
            )
            semantic_chunks = collect_semantic_chunks(source_refs, combined_text)
            selected_title = "Dokumen Kasus"
            if mode == "summary":
                formatted_answer, keypoints, summary = format_case_summary(
                    question,
                    combined_text,
                    semantic_chunks,
                )
            elif mode == "analysis":
                formatted_answer, keypoints, summary = format_case_analysis(
                    workspace_id,
                    question,
                    combined_text,
                    semantic_chunks,
                    source_refs,
                )
            else:
                formatted_answer, keypoints, summary = format_social_graph(
                    workspace_id,
                    question,
                    combined_text,
                    semantic_chunks,
                    source_refs,
                )
        else:
            combined_text, source_refs = collect_subsection_across_pages(
                workspace_id,
                best,
                selected_title,
                selected_text,
            )
            semantic_chunks = collect_semantic_chunks(source_refs, combined_text)
            graph_text, graph_source_refs = collect_document_pages(
                workspace_id,
                best.metadata.get("source", ""),
                docs,
            )
            graph_actors, graph_relationships = build_social_relationships(graph_text)
            graph_profiles = build_actor_profiles(graph_actors, graph_text)
            graph_bundle = build_graph_bundle(
                workspace_id=workspace_id,
                question=question,
                semantic_chunks=collect_semantic_chunks(graph_source_refs, graph_text),
                actors=graph_actors,
                relationships=graph_relationships,
                actor_profiles=graph_profiles,
            )
            if graph_bundle.cypher_path_lines:
                formatted_answer, keypoints, summary = format_graph_path_answer(
                    question,
                    graph_bundle.cypher_path_lines,
                )
                combined_text = graph_text
                source_refs = graph_source_refs
                selected_title = "Jalur Relasi Kasus"
            else:
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
