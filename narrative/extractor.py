from __future__ import annotations

import re
from collections import Counter

from narrative.classifier import DATE_REGEX
from narrative.ollama_client import OllamaClient, parse_json_list
from narrative.schemas import ClassifiedChunk, EvidenceLink, NarrativeEvent


ACTION_PATTERNS = [
    (r"(?i)\bmelaporkan\b|\bdilaporkan\b", "melaporkan"),
    (r"(?i)\bmenuduh\b|\bdidakwa\b|\bdiduga\b", "menuduh"),
    (r"(?i)\bmenghubungi\b|\bmenelepon\b|\bberkomunikasi\b", "menghubungi"),
    (r"(?i)\bbertemu\b|\bmenemui\b", "bertemu"),
    (r"(?i)\bmentransfer\b|\bmengirim\b|\bmenyetorkan\b", "mentransfer"),
    (r"(?i)\bmenerima\b|\bditerima\b", "menerima"),
    (r"(?i)\bmembayar\b|\bdibayar\b", "membayar"),
    (r"(?i)\bmenyerahkan\b|\bdiserahkan\b", "menyerahkan"),
    (r"(?i)\bmemeriksa\b|\bdiperiksa\b", "memeriksa"),
]

ACTOR_PATTERN = re.compile(
    r"\b(?:[A-Z][a-zA-Z.'-]+(?:\s+[A-Z][a-zA-Z.'-]+){0,4}|"
    r"[A-Z]{2,}(?:\s+[A-Z]{2,}){0,4}|"
    r"(?:Terdakwa|Korban|Saksi|Pemohon|Termohon|Penggugat|Tergugat|PT|CV|Bank)"
    r"(?:\s+[A-Z][a-zA-Z.'-]+){0,4})\b"
)

LOCATION_REGEX = re.compile(
    r"(?i)\b(?:di|bertempat di)\s+([A-Z][A-Za-z0-9.'\-/ ]{3,80}?)(?=,|\.|\s+pada|\s+tanggal|$)"
)


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    if not normalized:
        return []
    return [part.strip(" .:-") for part in re.split(r"(?<=[.!?])\s+|;\s+", normalized) if part.strip(" .:-")]


def normalize_actor(actor: str) -> str:
    cleaned = re.sub(r"\s+", " ", actor or "").strip(" .,:;()[]")
    cleaned = re.sub(r"^(Bahwa|Pada|Kemudian|Selanjutnya|Dalam)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    stop = {
        "Bahwa", "Pada", "Kemudian", "Selanjutnya", "Dalam", "Majelis", "Hakim",
        "Januari", "Februari", "Maret", "April", "Mei", "Juni", "Juli",
        "Agustus", "September", "Oktober", "November", "Desember",
    }
    return "" if cleaned in stop else cleaned


def extract_actors(text: str) -> list[str]:
    actors = []
    seen = set()
    for match in ACTOR_PATTERN.findall(text or ""):
        actor = normalize_actor(match)
        if len(actor) < 3 or actor.lower() in seen:
            continue
        seen.add(actor.lower())
        actors.append(actor)
        if len(actors) >= 6:
            break
    return actors


def detect_action(text: str) -> str:
    for pattern, action in ACTION_PATTERNS:
        if re.search(pattern, text):
            return action
    return "terkait"


def extract_target(sentence: str, actors: list[str], action: str) -> str:
    if len(actors) >= 2:
        return actors[1]
    if action == "terkait":
        return ""
    match = re.search(rf"(?i)\b{re.escape(action)}\b\s+(.+?)(?=,|\.|\s+pada|\s+di\s+|$)", sentence)
    return re.sub(r"\s+", " ", match.group(1)).strip(" .,:;") if match else ""


def extract_time(sentence: str) -> str:
    match = DATE_REGEX.search(sentence or "")
    if match:
        return match.group(0)
    lowered = sentence.lower()
    for marker in ("sebelumnya", "kemudian", "selanjutnya", "setelah itu", "hari yang sama", "akhirnya"):
        if marker in lowered:
            return marker
    return ""


def extract_location(sentence: str) -> str:
    match = LOCATION_REGEX.search(sentence or "")
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip(" .,:;")


def causal_marker(sentence: str) -> str:
    lowered = sentence.lower()
    for marker in ("karena", "sehingga", "akibatnya", "oleh karena itu", "dengan demikian"):
        if marker in lowered:
            return marker
    return ""


def token_overlap(left: str, right: str) -> float:
    left_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", left or "") if len(token) > 3}
    right_tokens = {token.lower() for token in re.findall(r"[A-Za-z0-9]+", right or "") if len(token) > 3}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / max(len(left_tokens), 1)


def link_evidence(sentence: str, chunk: ClassifiedChunk, evidence_chunks: list[ClassifiedChunk], limit: int = 3) -> list[EvidenceLink]:
    scored: list[tuple[float, EvidenceLink]] = []
    for evidence in evidence_chunks:
        page_distance = abs((evidence.page or 0) - (chunk.page or 0))
        proximity = max(0.0, 1.0 - (page_distance * 0.18))
        overlap = token_overlap(sentence, evidence.text)
        score = (overlap * 0.65) + (proximity * 0.35)
        if score < 0.2:
            continue
        excerpt = re.sub(r"\s+", " ", evidence.text).strip()
        scored.append((
            score,
            EvidenceLink(
                evidence_id=f"evd-{evidence.chunk_id}",
                chunk_id=evidence.chunk_id,
                page=evidence.page,
                text=excerpt[:260],
                relevance=round(score, 3),
                reason="token overlap and page proximity",
            ),
        ))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item for _, item in scored[:limit]]


class EventExtractor:
    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama or OllamaClient(timeout=50)

    async def extract_events(
        self,
        chunk: ClassifiedChunk,
        evidence_chunks: list[ClassifiedChunk],
    ) -> list[NarrativeEvent]:
        llm_events = await self._extract_with_llm(chunk, evidence_chunks)
        if llm_events:
            return llm_events
        return self.extract_events_heuristic(chunk, evidence_chunks)

    async def _extract_with_llm(
        self,
        chunk: ClassifiedChunk,
        evidence_chunks: list[ClassifiedChunk],
    ) -> list[NarrativeEvent]:
        if not self.ollama.enabled:
            return []

        evidence_context = "\n".join(
            f"- {item.chunk_id} p{item.page + 1}: {item.text[:280]}"
            for item in evidence_chunks[:6]
        )
        prompt = (
            "Ekstrak event dari narasi hukum/investigasi Indonesia. Jangan mengarang. "
            "Jika waktu/lokasi/sebab tidak jelas, isi string kosong dan tulis uncertainty.\n"
            "Jawab JSON saja berbentuk {\"events\":[...]}. Setiap event punya: "
            "actors(list), action, target, time, location, causal_relation, "
            "related_evidence(list chunk_id), confidence, uncertainty.\n\n"
            f"NARASI chunk_id={chunk.chunk_id} page={chunk.page + 1}:\n{chunk.text[:3200]}\n\n"
            f"BUKTI TERSEDIA:\n{evidence_context}"
        )
        response = await self.ollama.generate(prompt)
        parsed_events = parse_json_list(response)
        events: list[NarrativeEvent] = []
        for index, item in enumerate(parsed_events[:8], start=1):
            actors = [str(actor).strip() for actor in item.get("actors", []) if str(actor).strip()]
            action = str(item.get("action", "")).strip() or "terkait"
            related = [str(ref).strip() for ref in item.get("related_evidence", []) if str(ref).strip()]
            try:
                confidence = float(item.get("confidence", 0.65))
            except Exception:
                confidence = 0.65
            events.append(
                NarrativeEvent(
                    event_id=f"evt-{chunk.chunk_id}-{index}",
                    actors=actors[:6],
                    action=action,
                    target=str(item.get("target", "")).strip(),
                    time=str(item.get("time", "")).strip(),
                    location=str(item.get("location", "")).strip(),
                    causal_relation=str(item.get("causal_relation", "")).strip(),
                    related_evidence=related[:4],
                    confidence=max(0.0, min(1.0, confidence)),
                    source_chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    text=chunk.text[:500],
                    uncertainty=str(item.get("uncertainty", "")).strip(),
                )
            )
        return events

    def extract_events_heuristic(
        self,
        chunk: ClassifiedChunk,
        evidence_chunks: list[ClassifiedChunk],
    ) -> list[NarrativeEvent]:
        events: list[NarrativeEvent] = []
        for index, sentence in enumerate(split_sentences(chunk.text), start=1):
            actors = extract_actors(sentence)
            action = detect_action(sentence)
            time = extract_time(sentence)
            if not actors and action == "terkait" and not time:
                continue
            evidence_links = link_evidence(sentence, chunk, evidence_chunks)
            confidence = 0.42
            confidence += 0.18 if actors else 0.0
            confidence += 0.16 if action != "terkait" else 0.0
            confidence += 0.12 if time else 0.0
            confidence += 0.08 if evidence_links else 0.0
            events.append(
                NarrativeEvent(
                    event_id=f"evt-{chunk.chunk_id}-{index}",
                    actors=actors[:6],
                    action=action,
                    target=extract_target(sentence, actors, action),
                    time=time,
                    location=extract_location(sentence),
                    causal_relation=causal_marker(sentence),
                    related_evidence=[item.evidence_id for item in evidence_links],
                    confidence=round(min(confidence, 0.92), 3),
                    source_chunk_id=chunk.chunk_id,
                    page=chunk.page,
                    text=sentence,
                    uncertainty="" if time else "timestamp tidak eksplisit",
                )
            )
        return events[:8]


def top_actors(events: list[NarrativeEvent], limit: int = 14) -> list[str]:
    counter: Counter[str] = Counter()
    for event in events:
        for actor in event.actors:
            counter[actor] += 1
    return [actor for actor, _ in counter.most_common(limit)]
