from __future__ import annotations

import re

from rag_agent.models import ExtractionResult, ExtractedRelation


ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z0-9_-]+(?:\s+[A-Z][a-zA-Z0-9_-]+)*)\b")


def extract_entities_and_relations(text: str) -> ExtractionResult:
    entities = sorted({match.group(1) for match in ENTITY_PATTERN.finditer(text)})
    relations: list[ExtractedRelation] = []
    if len(entities) >= 2:
        relations.append(
            ExtractedRelation(
                subject=entities[0],
                predicate="mentions",
                obj=entities[1],
                evidence=text[:200],
            )
        )
    return ExtractionResult(entities=entities, relations=relations)
