from rag_agent.ingest.chunker import chunk_text


def test_chunk_text_respects_overlap_boundaries() -> None:
    text = " ".join(f"token{i}" for i in range(12))
    chunks = chunk_text(text, chunk_size=5, overlap=2)
    assert chunks == [
        "token0 token1 token2 token3 token4",
        "token3 token4 token5 token6 token7",
        "token6 token7 token8 token9 token10",
        "token9 token10 token11",
    ]
