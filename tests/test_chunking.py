from rag_agent.ingest.chunker import chunk_text


def test_chunk_text_respects_overlap_and_bounds() -> None:
    text = " ".join(f"token-{index}" for index in range(15))

    chunks = chunk_text(text, chunk_size=6, overlap=2)

    assert [chunk.start for chunk in chunks] == [0, 4, 8, 12]
    assert [chunk.end for chunk in chunks] == [6, 10, 14, 15]
    assert chunks[1].text.startswith("token-4 token-5")
