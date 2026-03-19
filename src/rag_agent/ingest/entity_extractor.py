from __future__ import annotations

from rag_agent.models import ExtractedRelation, ExtractionResult


RELATION_MARKERS = {
    "uses": "uses",
    "depends": "depends_on",
    "connects": "connects_to",
}


def extract_entities_and_relations(text: str) -> ExtractionResult:
    words = [word.strip('.,:;!?()[]{}') for word in text.split()]
    entities = sorted({word for word in words if word and word[:1].isupper()})
    relations: list[ExtractedRelation] = []
    lowered = [word.lower() for word in words]
    for marker, predicate in RELATION_MARKERS.items():
        if marker in lowered:
            idx = lowered.index(marker)
            if 0 < idx < len(words) - 1:
                relations.append(
                    ExtractedRelation(
                        subject=words[idx - 1],
                        predicate=predicate,
                        obj=words[idx + 1],
                        evidence=text,
                    )
                )
    return ExtractionResult(entities=entities, relations=relations)
