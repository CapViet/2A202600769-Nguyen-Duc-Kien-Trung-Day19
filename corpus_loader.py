"""
corpus_loader.py
================
Load + clean + chunk the lecturer's *US Electric Vehicle* corpus
(dataset/doc_1.txt ... doc_70.txt).

Each file looks like:
    Query:   <search query the doc was retrieved for>
    Title:   <article title>
    Link:    <url>
    Snippet: <short snippet>

    Full Content:
    <the article body -- web text, sometimes PDF-derived binary noise>

We parse those fields, strip web boilerplate (cookie/nav junk) and non-printable
PDF garbage, then chunk the body for retrieval. A *sampled* subset of leading
chunks per document is used for the (slow) LLM graph extraction.
"""
import glob
import os
import re

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

# lines containing any of these are dropped as boilerplate
_BOILER = re.compile(
    r"cookie|privacy policy|mailing list|subscribe|all rights reserved|"
    r"terms of use|sign up|newsletter|advertis|^\s*share\s*$|^\s*download\s*$|"
    r"follow us|©|use cookies|browser settings|google analytics",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    """Drop boilerplate lines and non-printable runs (PDF binary noise)."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or _BOILER.search(line):
            continue
        # ratio of printable ASCII; PDF-derived garbage is mostly non-ASCII/control
        printable = sum(1 for c in line if 32 <= ord(c) < 127)
        if len(line) and printable / len(line) < 0.85:
            continue
        if len(line) < 3:
            continue
        out.append(line)
    cleaned = "\n".join(out)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned


def load_documents(dataset_dir: str = DATASET_DIR):
    """Return list of dicts: {id, query, title, link, snippet, body}."""
    docs = []
    paths = sorted(
        glob.glob(os.path.join(dataset_dir, "doc_*.txt")),
        key=lambda p: int(re.search(r"doc_(\d+)", p).group(1)),
    )
    for p in paths:
        raw = open(p, encoding="utf-8", errors="ignore").read()
        def field(name):
            m = re.search(rf"^{name}:\s*(.+)$", raw, re.MULTILINE)
            return m.group(1).strip() if m else ""
        body = ""
        m = re.search(r"Full Content:\s*(.*)", raw, re.DOTALL)
        if m:
            body = _clean_text(m.group(1))
        docs.append({
            "id": os.path.splitext(os.path.basename(p))[0],
            "query": field("Query"),
            "title": field("Title"),
            "link": field("Link"),
            "snippet": _clean_text(field("Snippet")),
            "body": body,
        })
    return docs


def chunk_text(text: str, size: int = 1000, overlap: int = 150):
    """Split text into ~`size`-char chunks on sentence-ish boundaries with overlap."""
    text = text.strip()
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):  # try to end on a sentence boundary
            dot = text.rfind(". ", start + int(size * 0.6), end)
            if dot != -1:
                end = dot + 1
        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(docs, size=1000, overlap=150, per_doc=None, stride=False):
    """Flatten docs into retrieval passages.

    Each passage is prefixed with its source Title so retrieval/extraction keep
    document context. Returns list of dicts {doc_id, title, text}.

    `per_doc` caps chunks per doc (bounds graph-extraction cost). When
    `stride=True` the kept chunks are spread EVENLY across the document instead
    of just the leading ones -- important here because key facts (tickers, HQs,
    CEOs) are scattered through long web articles.
    """
    passages = []
    for d in docs:
        base = d["body"] if d["body"] else d["snippet"]
        ch = chunk_text(base, size=size, overlap=overlap)
        if per_doc is not None and len(ch) > per_doc:
            if stride:
                idx = sorted({round(i * (len(ch) - 1) / (per_doc - 1)) for i in range(per_doc)})
                ch = [ch[i] for i in idx]
            else:
                ch = ch[:per_doc]
        for c in ch:
            prefix = f"[{d['title'][:80]}] " if d["title"] else ""
            passages.append({"doc_id": d["id"], "title": d["title"], "text": prefix + c})
    return passages


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    full = build_chunks(docs, size=900, overlap=120)
    sampled = build_chunks(docs, size=900, overlap=120, per_doc=5, stride=True)
    print(f"Full chunks: {len(full)} | Sampled (stride 5/doc): {len(sampled)}")
    bodies = [len(d["body"]) for d in docs]
    print(f"Body chars: total={sum(bodies):,} | max={max(bodies):,} | empty bodies={sum(1 for b in bodies if b==0)}")
    print("\nExample sampled passage:\n", sampled[0]["text"][:300])
