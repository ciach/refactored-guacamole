from __future__ import annotations

from pathlib import Path

from rag_agent.compat import PdfReader, parse_frontmatter


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf"}


def load_document(path: Path) -> tuple[str, str]:
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document type: {path.suffix}")
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return path.stem, text.strip()
    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".md", ".markdown"}:
        metadata, content = parse_frontmatter(raw_text)
        title = str(metadata.get("title", path.stem))
        return title, content
    return path.stem, raw_text.strip()
