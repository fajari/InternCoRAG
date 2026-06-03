from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ChunkClassification = Literal[
    "narrative",
    "evidence",
    "metadata",
    "legal_reference",
    "noise",
]


@dataclass
class ClassifiedChunk:
    chunk_id: str
    page: int
    text: str
    section: str
    classification: ChunkClassification
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceLink:
    evidence_id: str
    chunk_id: str
    page: int
    text: str
    relevance: float
    reason: str = ""


@dataclass
class NarrativeEvent:
    event_id: str
    actors: list[str]
    action: str
    target: str
    time: str
    location: str
    causal_relation: str
    related_evidence: list[str]
    confidence: float
    source_chunk_id: str
    page: int
    text: str
    uncertainty: str = ""


@dataclass
class StoryGraph:
    actor_relations: list[tuple[str, str, str, str]]
    event_relations: list[tuple[str, str, str]]
    evidence_relations: list[tuple[str, str, str]]
    temporal_relations: list[tuple[str, str, str]]
