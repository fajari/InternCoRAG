from __future__ import annotations

import re
from dataclasses import dataclass

from config import NARRATIVE_LLM_CLASSIFIER_THRESHOLD
from narrative.ollama_client import OllamaClient, parse_json_object
from narrative.schemas import ChunkClassification


CLASSIFICATIONS: set[str] = {
    "narrative",
    "evidence",
    "metadata",
    "legal_reference",
    "noise",
}

NARRATIVE_TERMS = {
    "bahwa", "kemudian", "selanjutnya", "setelah", "sebelum", "pada tanggal",
    "saksi", "terdakwa", "korban", "pemohon", "termohon", "penggugat",
    "tergugat", "menghubungi", "bertemu", "menerima", "menyerahkan",
    "mentransfer", "membayar", "melaporkan", "peristiwa", "kejadian",
}
EVIDENCE_TERMS = {
    "bukti", "barang bukti", "rekening", "mutasi", "transfer", "invoice",
    "kwitansi", "nota", "hash", "log", "lampiran", "foto", "screenshot",
    "berita acara", "surat", "dokumen", "rekaman",
}
METADATA_TERMS = {
    "nomor perkara", "putusan nomor", "identitas", "nama:", "tempat lahir",
    "tanggal lahir", "umur", "jenis kelamin", "kebangsaan", "tempat tinggal",
    "agama", "pekerjaan", "klasifikasi", "status dokumen", "approved by",
}
LEGAL_TERMS = {
    "pasal", "undang-undang", "kuhp", "kuhap", "peraturan", "menimbang",
    "mengingat", "mengadili", "amar putusan", "yurisprudensi", "mahkamah agung",
    "putusan", "hakim", "jaksa", "penuntut umum",
}
NOISE_TERMS = {
    "halaman", "page", "www.", "email", "e-mail", "telp", "fax",
    "disclaimer", "untuk salinan yang sama bunyinya",
}

DATE_REGEX = re.compile(
    r"(?i)\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|"
    r"\d{1,2}\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|"
    r"oktober|november|desember)\s+\d{2,4})\b"
)


@dataclass
class ClassificationResult:
    classification: ChunkClassification
    confidence: float
    reason: str = ""


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def keyword_hits(text: str, terms: set[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term in lowered)


def looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    tableish = 0
    for line in lines:
        if "|" in line or "\t" in line or len(re.split(r"\s{2,}", line)) >= 3:
            tableish += 1
    return tableish >= max(2, len(lines) // 3)


def looks_like_noise(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if len(normalized) < 20:
        return True
    if re.fullmatch(r"[\W\d_]+", normalized):
        return True
    lowered = normalized.lower()
    if keyword_hits(lowered, NOISE_TERMS) >= 2 and len(normalized.split()) <= 20:
        return True
    return False


def heuristic_classify(text: str) -> ClassificationResult:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if looks_like_noise(normalized):
        return ClassificationResult("noise", 0.86, "short/noisy/footer-like text")

    narrative = keyword_hits(normalized, NARRATIVE_TERMS)
    evidence = keyword_hits(normalized, EVIDENCE_TERMS)
    metadata = keyword_hits(normalized, METADATA_TERMS)
    legal = keyword_hits(normalized, LEGAL_TERMS)
    has_date = bool(DATE_REGEX.search(normalized))
    table = looks_like_table(text)
    has_hash = bool(re.search(r"\b[a-f0-9]{32,64}\b", normalized, flags=re.IGNORECASE))
    has_money = bool(re.search(r"(?i)\b(rp|idr)\s*[\d.,]+|\b\d{1,3}(?:\.\d{3})+(?:,\d+)?\b", normalized))

    scores = {
        "narrative": narrative * 0.18 + (0.22 if has_date else 0.0),
        "evidence": evidence * 0.18 + (0.22 if table else 0.0) + (0.18 if has_hash or has_money else 0.0),
        "metadata": metadata * 0.2,
        "legal_reference": legal * 0.16,
        "noise": 0.0,
    }

    if legal >= 3 and narrative <= 1:
        scores["legal_reference"] += 0.35
    if metadata >= 4 and narrative <= 1:
        scores["metadata"] += 0.35
    if table and narrative <= 1:
        scores["evidence"] += 0.2
    if narrative >= 2 and has_date:
        scores["narrative"] += 0.25

    label = max(scores, key=scores.get)
    confidence = clamp_confidence(0.48 + scores[label])
    return ClassificationResult(label, confidence, f"heuristic scores={scores}")


class HybridNarrativeClassifier:
    def __init__(self, ollama: OllamaClient | None = None, llm_threshold: float = 0.72):
        self.ollama = ollama or OllamaClient(timeout=25)
        self.llm_threshold = llm_threshold if llm_threshold != 0.72 else NARRATIVE_LLM_CLASSIFIER_THRESHOLD

    async def classify(self, text: str) -> ClassificationResult:
        heuristic = heuristic_classify(text)
        if heuristic.confidence >= self.llm_threshold or not self.ollama.enabled:
            return heuristic

        prompt = (
            "Klasifikasikan potongan dokumen hukum/investigasi Indonesia berikut.\n"
            "Pilih tepat satu: narrative, evidence, metadata, legal_reference, noise.\n"
            "Definisi singkat: narrative=cerita/peristiwa dengan aktor/aksi/waktu/sebab; "
            "evidence=bukti, angka, transaksi, tabel, hash, log, lampiran; "
            "metadata=identitas/form administrasi; legal_reference=pasal, amar, dasar hukum; "
            "noise=header/footer/OCR rusak.\n"
            "Jawab JSON saja: {\"classification\":\"...\",\"confidence\":0.0,\"reason\":\"...\"}\n\n"
            f"TEKS:\n{text[:2500]}"
        )
        response = await self.ollama.generate(prompt)
        parsed = parse_json_object(response)
        label = str(parsed.get("classification", "")).strip().lower()
        if label not in CLASSIFICATIONS:
            return heuristic
        try:
            llm_confidence = float(parsed.get("confidence", heuristic.confidence))
        except Exception:
            llm_confidence = heuristic.confidence
        confidence = clamp_confidence((heuristic.confidence + llm_confidence) / 2)
        return ClassificationResult(label, confidence, str(parsed.get("reason", "ollama classifier")))
