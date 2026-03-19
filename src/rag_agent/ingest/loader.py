from __future__ import annotations

from pathlib import Path


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        try:  # pragma: no cover - requires pypdf.
            from pypdf import PdfReader
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Install pypdf to load PDFs.") from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported document type: {path.suffix}")
