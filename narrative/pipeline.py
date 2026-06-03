from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

from narrative.classifier import HybridNarrativeClassifier
from narrative.extractor import EventExtractor, top_actors
from narrative.ollama_client import OllamaClient, parse_json_object
from narrative.schemas import ClassifiedChunk, EvidenceLink, NarrativeEvent, StoryGraph


@dataclass
class NarrativeAnalysis:
    chunks: list[ClassifiedChunk]
    narrative_chunks: list[ClassifiedChunk]
    evidence_chunks: list[ClassifiedChunk]
    metadata_chunks: list[ClassifiedChunk]
    legal_reference_chunks: list[ClassifiedChunk]
    noise_chunks: list[ClassifiedChunk]
    events: list[NarrativeEvent]
    evidence_links: list[EvidenceLink]
    story_graph: StoryGraph
    chunk_summaries: dict[str, str] = field(default_factory=dict)
    section_summaries: dict[str, str] = field(default_factory=dict)
    narrative_summary: str = ""
    executive_summary: str = ""


def get_ref_content(item: Any) -> str:
    return item.get("page_content", "") if isinstance(item, dict) else getattr(item, "page_content", "")


def get_ref_metadata(item: Any) -> dict[str, Any]:
    return item.get("metadata", {}) if isinstance(item, dict) else getattr(item, "metadata", {})


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def build_chunk_id(index: int, metadata: dict[str, Any]) -> str:
    source = re.sub(r"[^A-Za-z0-9]+", "-", str(metadata.get("source", "doc"))).strip("-").lower() or "doc"
    page = metadata.get("page", 0)
    semantic_index = metadata.get("semantic_chunk_index", index)
    return f"{source}-p{int(page) + 1}-c{semantic_index}"


def sort_event_key(event: NarrativeEvent, index: int) -> tuple[int, int, int]:
    normalized = (event.time or "").lower()
    match = re.search(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})", normalized)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    match = re.search(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", normalized)
    if match:
        year = int(match.group(3))
        if year < 100:
            year += 2000
        return year, int(match.group(2)), int(match.group(1))
    relative_rank = {
        "sebelumnya": 10,
        "hari yang sama": 20,
        "kemudian": 30,
        "selanjutnya": 40,
        "setelah itu": 50,
        "akhirnya": 60,
    }
    for marker, rank in relative_rank.items():
        if marker in normalized:
            return 9998, 0, rank
    return 9999, event.page, index


def sentence_summary(text: str, max_words: int = 28) -> str:
    normalized = normalize_text(text).strip(" .:-")
    if not normalized:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+|;\s+", normalized)[0].strip(" .:-")
    words = sentence.split()
    if len(words) > max_words:
        return " ".join(words[:max_words]).rstrip(",.:") + "..."
    return sentence


def event_to_sentence(event: NarrativeEvent) -> str:
    pieces = []
    if event.time:
        pieces.append(f"Pada {event.time}")
    if event.actors:
        pieces.append(", ".join(event.actors[:3]))
    if event.action:
        pieces.append(event.action)
    if event.target:
        pieces.append(event.target)
    if event.location:
        pieces.append(f"di {event.location}")
    if event.uncertainty:
        pieces.append(f"(catatan: {event.uncertainty})")
    statement = " ".join(piece for piece in pieces if piece).strip()
    return statement or sentence_summary(event.text)


def build_story_graph(events: list[NarrativeEvent]) -> StoryGraph:
    actor_relations: list[tuple[str, str, str, str]] = []
    event_relations: list[tuple[str, str, str]] = []
    evidence_relations: list[tuple[str, str, str]] = []
    temporal_relations: list[tuple[str, str, str]] = []
    seen_actor_edges = set()

    ordered = sorted(enumerate(events), key=lambda item: sort_event_key(item[1], item[0]))
    for order_index, (_, event) in enumerate(ordered):
        if order_index > 0:
            temporal_relations.append((ordered[order_index - 1][1].event_id, "sebelum", event.event_id))

        if event.causal_relation and order_index > 0:
            event_relations.append((ordered[order_index - 1][1].event_id, event.causal_relation, event.event_id))

        for evidence_id in event.related_evidence:
            evidence_relations.append((event.event_id, "didukung_bukti", evidence_id))

        if len(event.actors) >= 2:
            left, right = event.actors[0], event.actors[1]
            key = (left.lower(), event.action.lower(), right.lower())
            if key not in seen_actor_edges:
                seen_actor_edges.add(key)
                actor_relations.append((left, event.action or "terkait", right, sentence_summary(event.text, 18)))
        elif event.actors and event.target:
            left, right = event.actors[0], event.target
            key = (left.lower(), event.action.lower(), right.lower())
            if key not in seen_actor_edges:
                seen_actor_edges.add(key)
                actor_relations.append((left, event.action or "terkait", right, sentence_summary(event.text, 18)))

    return StoryGraph(
        actor_relations=actor_relations[:24],
        event_relations=event_relations[:24],
        evidence_relations=evidence_relations[:48],
        temporal_relations=temporal_relations[:48],
    )


class HierarchicalSummarizer:
    def __init__(self, ollama: OllamaClient | None = None):
        self.ollama = ollama or OllamaClient(timeout=70)

    async def summarize(self, chunks: list[ClassifiedChunk], events: list[NarrativeEvent]) -> tuple[dict[str, str], dict[str, str], str, str]:
        chunk_summaries: dict[str, str] = {}
        for chunk in chunks[:16]:
            chunk_summaries[chunk.chunk_id] = await self.summarize_chunk(chunk)

        section_summaries: dict[str, str] = {}
        for section in sorted({chunk.section or "Dokumen Kasus" for chunk in chunks}):
            section_items = [chunk_summaries.get(chunk.chunk_id, "") for chunk in chunks if (chunk.section or "Dokumen Kasus") == section]
            seed = "\n".join(item for item in section_items if item)
            if seed:
                section_summaries[section] = await self.summarize_text(seed, "Buat ringkasan section kronologis, hanya dari klaim yang ada.")

        event_lines = [event_to_sentence(event) for _, event in sorted(enumerate(events), key=lambda item: sort_event_key(item[1], item[0]))]
        narrative_seed = "\n".join(f"- {line}" for line in event_lines if line)
        narrative_summary = await self.summarize_text(
            narrative_seed or "\n".join(chunk_summaries.values()),
            "Buat rangkuman narasi kronologis Bahasa Indonesia. Jangan mengarang. Tandai ketidakpastian.",
        )
        executive_summary = await self.summarize_text(
            narrative_summary,
            "Buat executive summary sangat singkat, kronologis, dan traceable.",
        )
        return chunk_summaries, section_summaries, narrative_summary, executive_summary

    async def summarize_chunk(self, chunk: ClassifiedChunk) -> str:
        return await self.summarize_text(
            chunk.text[:2600],
            f"Ringkas chunk narasi ini dalam Bahasa Indonesia. Cantumkan jejak [{chunk.chunk_id} p{chunk.page + 1}]. Jangan menambah fakta.",
        )

    async def summarize_text(self, text: str, instruction: str) -> str:
        text = normalize_text(text)
        if not text:
            return ""
        if self.ollama.enabled:
            prompt = (
                f"{instruction}\n"
                "Semua klaim harus berasal dari teks. Jika tidak pasti, tulis 'tidak pasti'.\n"
                "Jawab JSON saja: {\"summary\":\"...\"}\n\n"
                f"TEKS:\n{text[:4500]}"
            )
            response = await self.ollama.generate(prompt)
            parsed = parse_json_object(response)
            summary = normalize_text(str(parsed.get("summary", "")))
            if summary:
                return summary
        return sentence_summary(text, max_words=55)


class NarrativePipeline:
    def __init__(
        self,
        classifier: HybridNarrativeClassifier | None = None,
        extractor: EventExtractor | None = None,
        summarizer: HierarchicalSummarizer | None = None,
    ):
        ollama = OllamaClient()
        self.classifier = classifier or HybridNarrativeClassifier(ollama)
        self.extractor = extractor or EventExtractor(ollama)
        self.summarizer = summarizer or HierarchicalSummarizer(ollama)

    async def analyze(self, source_refs: list[Any], fallback_text: str = "") -> NarrativeAnalysis:
        chunks = await self.classify_refs(source_refs)
        if not chunks and fallback_text:
            result = await self.classifier.classify(fallback_text)
            chunks = [
                ClassifiedChunk(
                    chunk_id="fallback-p1-c0",
                    page=0,
                    text=fallback_text,
                    section="Dokumen Kasus",
                    classification=result.classification,
                    confidence=result.confidence,
                    metadata={"classifier_reason": result.reason},
                )
            ]

        narrative_chunks = [chunk for chunk in chunks if chunk.classification == "narrative"]
        evidence_chunks = [chunk for chunk in chunks if chunk.classification == "evidence"]
        metadata_chunks = [chunk for chunk in chunks if chunk.classification == "metadata"]
        legal_reference_chunks = [chunk for chunk in chunks if chunk.classification == "legal_reference"]
        noise_chunks = [chunk for chunk in chunks if chunk.classification == "noise"]

        event_batches = await asyncio.gather(
            *[self.extractor.extract_events(chunk, evidence_chunks) for chunk in narrative_chunks[:18]]
        ) if narrative_chunks else []
        events = [event for batch in event_batches for event in batch]
        events = sorted(events, key=lambda event: sort_event_key(event, events.index(event) if event in events else 0))
        evidence_links = self.collect_evidence_links(events, evidence_chunks)
        story_graph = build_story_graph(events)
        chunk_summaries, section_summaries, narrative_summary, executive_summary = await self.summarizer.summarize(narrative_chunks, events)

        return NarrativeAnalysis(
            chunks=chunks,
            narrative_chunks=narrative_chunks,
            evidence_chunks=evidence_chunks,
            metadata_chunks=metadata_chunks,
            legal_reference_chunks=legal_reference_chunks,
            noise_chunks=noise_chunks,
            events=events,
            evidence_links=evidence_links,
            story_graph=story_graph,
            chunk_summaries=chunk_summaries,
            section_summaries=section_summaries,
            narrative_summary=narrative_summary,
            executive_summary=executive_summary,
        )

    async def classify_refs(self, source_refs: list[Any]) -> list[ClassifiedChunk]:
        chunks: list[ClassifiedChunk] = []
        for index, item in enumerate(source_refs):
            text = normalize_text(get_ref_content(item))
            if not text:
                continue
            metadata = dict(get_ref_metadata(item))
            existing = str(metadata.get("classification") or metadata.get("content_type") or "").strip().lower()
            if existing in {"narrative", "evidence", "metadata", "legal_reference", "noise"}:
                classification = existing
                confidence = float(metadata.get("classification_confidence", 0.74))
                reason = "metadata classification"
            else:
                result = await self.classifier.classify(text)
                classification = result.classification
                confidence = result.confidence
                reason = result.reason
            chunk_id = str(metadata.get("chunk_id") or build_chunk_id(index, metadata))
            chunks.append(
                ClassifiedChunk(
                    chunk_id=chunk_id,
                    page=int(metadata.get("page", 0) or 0),
                    text=text,
                    section=str(metadata.get("section", "Dokumen Kasus")),
                    classification=classification,  # type: ignore[arg-type]
                    confidence=confidence,
                    metadata={**metadata, "classifier_reason": reason},
                )
            )
        return chunks

    def collect_evidence_links(
        self,
        events: list[NarrativeEvent],
        evidence_chunks: list[ClassifiedChunk],
    ) -> list[EvidenceLink]:
        evidence_by_id = {
            f"evd-{chunk.chunk_id}": chunk
            for chunk in evidence_chunks
        }
        links: list[EvidenceLink] = []
        seen = set()
        for event in events:
            for evidence_id in event.related_evidence:
                chunk = evidence_by_id.get(evidence_id)
                if not chunk or evidence_id in seen:
                    continue
                seen.add(evidence_id)
                links.append(
                    EvidenceLink(
                        evidence_id=evidence_id,
                        chunk_id=chunk.chunk_id,
                        page=chunk.page,
                        text=chunk.text[:260],
                        relevance=0.75,
                        reason=f"linked from {event.event_id}",
                    )
                )
        return links


def analyze_documents_sync(source_refs: list[Any], fallback_text: str = "") -> NarrativeAnalysis:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(NarrativePipeline().analyze(source_refs, fallback_text))

    # Streamlit normally runs synchronously. If a loop exists, isolate the async
    # pipeline in a worker thread to keep the public integration sync-friendly.
    result: list[NarrativeAnalysis] = []

    def runner() -> None:
        result.append(asyncio.run(NarrativePipeline().analyze(source_refs, fallback_text)))

    import threading

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()
    return result[0]
