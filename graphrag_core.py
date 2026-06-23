"""
graphrag_core.py
================
Core logic for the Day-19 GraphRAG lab. Kept as a plain module so the pieces
can be unit-tested quickly against Ollama before being mirrored into the
notebook cells. Fully local stack:

    LLM        : Ollama (qwen2.5:3b) via HTTP  -> extraction + answer generation
    Embeddings : sentence-transformers all-MiniLM-L6-v2 (USE_TF=0 to dodge TF/Keras3)
    Vector DB  : ChromaDB (Flat RAG)
    Graph      : NetworkX MultiDiGraph (GraphRAG)
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import re
import time
import urllib.request
from collections import defaultdict

OLLAMA_URL = "http://localhost:11434/api"
LLM_MODEL = "qwen2.5:3b"

# ---------------------------------------------------------------------------
# Token accounting -- lets us report real token usage / latency (deliverable 4)
# ---------------------------------------------------------------------------
STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}


def reset_stats():
    for k in STATS:
        STATS[k] = 0 if k != "seconds" else 0.0


def ollama_generate(prompt, system=None, temperature=0.0, num_predict=512, fmt=None):
    """Single-turn completion via Ollama; records token + latency stats."""
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if system:
        payload["system"] = system
    if fmt:
        payload["format"] = fmt  # e.g. "json" -> constrained JSON output
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OLLAMA_URL + "/generate", data=data, headers={"Content-Type": "application/json"}
    )
    t = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    STATS["calls"] += 1
    STATS["seconds"] += time.time() - t
    STATS["prompt_tokens"] += resp.get("prompt_eval_count", 0)
    STATS["completion_tokens"] += resp.get("eval_count", 0)
    return resp.get("response", "")


# ---------------------------------------------------------------------------
# Step 1 -- Entity & Relation extraction (Indexing)
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM = (
    "You are a precise information extraction engine. "
    "From the text you output a knowledge graph as SUBJECT-PREDICATE-OBJECT triples. "
    "A SUBJECT/OBJECT is a named entity (company, person, product, place) or a literal "
    "value such as a year. The PREDICATE is an UPPER_SNAKE_CASE relation. "
    "Return ONLY JSON of the form {\"triples\": [[\"subject\",\"PREDICATE\",\"object\"], ...]}. "
    "Use canonical entity names (e.g. 'Google' not 'the company'). Do not invent facts."
)

EXTRACT_FEWSHOT = (
    'Text: "OpenAI was founded by Sam Altman and Elon Musk in 2015. It created ChatGPT."\n'
    'JSON: {"triples": [["OpenAI","FOUNDED_BY","Sam Altman"],'
    '["OpenAI","FOUNDED_BY","Elon Musk"],["OpenAI","FOUNDED_IN","2015"],'
    '["OpenAI","CREATED","ChatGPT"]]}'
)


def extract_triples(text):
    """LLM -> list of (subj, pred, obj) triples for one document."""
    prompt = f"{EXTRACT_FEWSHOT}\n\nText: \"{text}\"\nJSON:"
    raw = ollama_generate(prompt, system=EXTRACT_SYSTEM, fmt="json", num_predict=512)
    triples = []
    try:
        obj = json.loads(raw)
        for t in obj.get("triples", []):
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, p, o = (str(x).strip() for x in t)
                if s and p and o:
                    triples.append((s, p.upper().replace(" ", "_"), o))
    except json.JSONDecodeError:
        # best-effort regex salvage if the model wrapped JSON in prose
        for m in re.finditer(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', raw):
            s, p, o = m.groups()
            triples.append((s.strip(), p.strip().upper().replace(" ", "_"), o.strip()))
    return triples


# ---------------------------------------------------------------------------
# Step 2 -- Graph construction with deduplication (normalisation)
# ---------------------------------------------------------------------------
def canonical(name):
    """Normalise an entity surface form so duplicates collapse to one node."""
    n = name.strip().strip('."\'')
    aliases = {
        "google llc": "Google", "google inc": "Google", "alphabet": "Alphabet",
        "openai inc": "OpenAI", "open ai": "OpenAI",
        "meta platforms": "Meta", "facebook inc": "Meta", "facebook": "Meta",
        "microsoft corporation": "Microsoft", "microsoft corp": "Microsoft",
        "deep mind": "DeepMind", "nvidia corporation": "Nvidia",
    }
    return aliases.get(n.lower(), n)


def build_graph(all_triples):
    """Return a NetworkX MultiDiGraph; nodes deduped via canonical()."""
    import networkx as nx

    G = nx.MultiDiGraph()
    seen = set()
    for s, p, o in all_triples:
        cs, co = canonical(s), canonical(o)
        G.add_node(cs)
        G.add_node(co)
        key = (cs, p, co)
        if key in seen:
            continue
        seen.add(key)
        G.add_edge(cs, co, relation=p)
    return G


# ---------------------------------------------------------------------------
# Step 3 -- GraphRAG query: entity linking -> 2-hop subgraph -> textualize -> LLM
# ---------------------------------------------------------------------------
def link_entities(question, G):
    """Find graph nodes mentioned in the question (case-insensitive substring)."""
    q = question.lower()
    hits = []
    for node in G.nodes():
        nl = str(node).lower()
        if len(nl) >= 3 and nl in q:
            hits.append(node)
    # prefer longer / more specific matches, drop nodes contained in another hit
    hits.sort(key=lambda n: len(str(n)), reverse=True)
    final = []
    for h in hits:
        if not any(h != o and str(h).lower() in str(o).lower() for o in final):
            final.append(h)
    return final


def k_hop_subgraph(G, seeds, k=2):
    """BFS up to k hops (following edges in both directions) from seed nodes."""
    import networkx as nx

    UG = G.to_undirected(as_view=True)
    nodes = set(seeds)
    frontier = set(seeds)
    for _ in range(k):
        nxt = set()
        for n in frontier:
            if n in UG:
                nxt |= set(UG.neighbors(n))
        nodes |= nxt
        frontier = nxt
    return nodes


def textualize(G, nodes):
    """Turn the retrieved subgraph edges into natural-language sentences."""
    PHRASES = {
        "FOUNDED_BY": "was founded by", "FOUNDED_IN": "was founded in",
        "CREATED": "created", "DEVELOPED": "developed", "ACQUIRED": "acquired",
        "ACQUIRED_BY": "was acquired by", "CEO_IS": "has CEO", "CEO": "has CEO",
        "HEADQUARTERED_IN": "is headquartered in", "INVESTED_IN": "invested in",
        "OWNS": "owns", "SUBSIDIARY_OF": "is a subsidiary of", "PARTNERED_WITH": "partnered with",
        "RELEASED": "released", "BASED_IN": "is based in",
    }
    nodeset = set(nodes)
    lines = []
    seen = set()
    for u, v, d in G.edges(data=True):
        if u in nodeset and v in nodeset:
            rel = d.get("relation", "RELATED_TO")
            phrase = PHRASES.get(rel, rel.lower().replace("_", " "))
            sent = f"{u} {phrase} {v}."
            if sent not in seen:
                seen.add(sent)
                lines.append(sent)
    return "\n".join(lines)


GRAPH_QA_SYSTEM = (
    "You answer the question using ONLY the facts in the provided knowledge-graph "
    "context. If the answer is not present, reply exactly 'I don't know.'. "
    "Answer in one concise sentence."
)


def graphrag_answer(question, G, k=2):
    seeds = link_entities(question, G)
    if not seeds:
        return "I don't know.", "", []
    nodes = k_hop_subgraph(G, seeds, k=k)
    context = textualize(G, nodes)
    prompt = f"Knowledge graph facts:\n{context}\n\nQuestion: {question}\nAnswer:"
    ans = ollama_generate(prompt, system=GRAPH_QA_SYSTEM, num_predict=128)
    return ans.strip(), context, seeds


# ---------------------------------------------------------------------------
# Flat RAG baseline -- MiniLM embeddings + FAISS (cosine via inner product)
# ---------------------------------------------------------------------------
FLAT_QA_SYSTEM = (
    "You answer the question using ONLY the provided text passages. "
    "If the answer is not present, reply exactly 'I don't know.'. "
    "Answer in one concise sentence."
)


class FlatRAG:
    """Dense-retrieval baseline: each corpus sentence is one passage, embedded
    with MiniLM and indexed in FAISS. Cosine similarity = inner product on
    L2-normalised vectors."""

    def __init__(self, docs, model_name="all-MiniLM-L6-v2"):
        import numpy as np
        import faiss
        from sentence_transformers import SentenceTransformer

        self.docs = docs
        self.model = SentenceTransformer(model_name)
        embs = self.model.encode(docs, show_progress_bar=False).astype("float32")
        faiss.normalize_L2(embs)
        self.index = faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def retrieve(self, question, k=4):
        import numpy as np
        import faiss

        q = self.model.encode([question]).astype("float32")
        faiss.normalize_L2(q)
        _, idx = self.index.search(q, k)
        return [self.docs[i] for i in idx[0] if i != -1]

    def answer(self, question, k=4):
        ctx = self.retrieve(question, k=k)
        passages = "\n".join(f"- {c}" for c in ctx)
        prompt = f"Passages:\n{passages}\n\nQuestion: {question}\nAnswer:"
        ans = ollama_generate(prompt, system=FLAT_QA_SYSTEM, num_predict=128)
        return ans.strip(), ctx
