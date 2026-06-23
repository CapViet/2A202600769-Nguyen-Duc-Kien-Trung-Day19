"""
graphrag_core.py
================
Core logic for the Day-19 GraphRAG lab.

LLM provider is **pluggable** -- switch backend with ONE environment variable,
no code change, so a grader can run the whole pipeline on their own provider:

    # local, free (default)
    GRAPHRAG_PROVIDER=ollama     GRAPHRAG_MODEL=qwen2.5:3b
    # OpenAI
    GRAPHRAG_PROVIDER=openai     GRAPHRAG_MODEL=gpt-4o-mini      OPENAI_API_KEY=...
    # Anthropic Claude
    GRAPHRAG_PROVIDER=anthropic  GRAPHRAG_MODEL=claude-opus-4-8  ANTHROPIC_API_KEY=...
    # Google Gemini
    GRAPHRAG_PROVIDER=google     GRAPHRAG_MODEL=gemini-1.5-flash GOOGLE_API_KEY=...

Default stack: Ollama LLM + MiniLM embeddings + FAISS (Flat RAG) + NetworkX (graph).
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

# ---------------------------------------------------------------------------
# Provider configuration (override via environment variables)
# ---------------------------------------------------------------------------
PROVIDER = os.environ.get("GRAPHRAG_PROVIDER", "ollama").lower()
_DEFAULT_MODELS = {
    "ollama": "qwen2.5:3b",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-opus-4-8",
    "google": "gemini-1.5-flash",
}
LLM_MODEL = os.environ.get("GRAPHRAG_MODEL", _DEFAULT_MODELS.get(PROVIDER, "qwen2.5:3b"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api")

# ---------------------------------------------------------------------------
# Token accounting -- lets us report real token usage / latency (deliverable 4)
# ---------------------------------------------------------------------------
STATS = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}


def reset_stats():
    for k in STATS:
        STATS[k] = 0 if k != "seconds" else 0.0


def _record(t0, ptok, ctok):
    STATS["calls"] += 1
    STATS["seconds"] += time.time() - t0
    STATS["prompt_tokens"] += ptok or 0
    STATS["completion_tokens"] += ctok or 0


def _gen_ollama(prompt, system, temperature, num_predict, fmt):
    payload = {
        "model": LLM_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if system:
        payload["system"] = system
    if fmt:
        payload["format"] = fmt  # "json" -> constrained JSON output
    req = urllib.request.Request(
        OLLAMA_URL + "/generate", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
    _record(t0, resp.get("prompt_eval_count"), resp.get("eval_count"))
    return resp.get("response", "")


def _gen_openai(prompt, system, temperature, num_predict, fmt):
    from openai import OpenAI
    client = OpenAI()
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    kw = {"model": LLM_MODEL, "messages": msgs, "temperature": temperature,
          "max_tokens": num_predict}
    if fmt == "json":
        kw["response_format"] = {"type": "json_object"}
    t0 = time.time()
    r = client.chat.completions.create(**kw)
    u = getattr(r, "usage", None)
    _record(t0, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0))
    return r.choices[0].message.content or ""


def _gen_anthropic(prompt, system, temperature, num_predict, fmt):
    import anthropic
    client = anthropic.Anthropic()
    if fmt == "json":
        prompt += "\n\nRespond with ONLY valid JSON."
    t0 = time.time()
    r = client.messages.create(
        model=LLM_MODEL, max_tokens=num_predict, temperature=temperature,
        system=system or anthropic.NOT_GIVEN,
        messages=[{"role": "user", "content": prompt}])
    _record(t0, r.usage.input_tokens, r.usage.output_tokens)
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def _gen_google(prompt, system, temperature, num_predict, fmt):
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    cfg = {"temperature": temperature, "max_output_tokens": num_predict}
    if fmt == "json":
        cfg["response_mime_type"] = "application/json"
    model = genai.GenerativeModel(LLM_MODEL, system_instruction=system,
                                  generation_config=cfg)
    t0 = time.time()
    r = model.generate_content(prompt)
    um = getattr(r, "usage_metadata", None)
    _record(t0, getattr(um, "prompt_token_count", 0),
            getattr(um, "candidates_token_count", 0))
    return r.text


_BACKENDS = {"ollama": _gen_ollama, "openai": _gen_openai,
             "anthropic": _gen_anthropic, "google": _gen_google}


def llm_generate(prompt, system=None, temperature=0.0, num_predict=512, fmt=None):
    """Single-turn completion via the configured PROVIDER; records token/latency
    stats. Switch provider with GRAPHRAG_PROVIDER / GRAPHRAG_MODEL env vars."""
    backend = _BACKENDS.get(PROVIDER)
    if backend is None:
        raise ValueError(f"Unknown GRAPHRAG_PROVIDER={PROVIDER!r}; "
                         f"choose one of {list(_BACKENDS)}")
    return backend(prompt, system, temperature, num_predict, fmt)


# Backwards-compatible alias (older call sites / notebooks)
ollama_generate = llm_generate


# ---------------------------------------------------------------------------
# Step 1 -- Entity & Relation extraction (Indexing)
# ---------------------------------------------------------------------------
EXTRACT_SYSTEM = (
    "You are a precise information-extraction engine for the ELECTRIC VEHICLE (EV) "
    "industry. From the text you output a knowledge graph as SUBJECT-PREDICATE-OBJECT "
    "triples. A SUBJECT/OBJECT is a named entity (EV company, person/executive, vehicle "
    "model, stock ticker, place, organization) or a literal value such as a year or "
    "percentage. The PREDICATE is an UPPER_SNAKE_CASE relation such as CEO_IS, FOUNDED_BY, "
    "FOUNDED_IN, HEADQUARTERED_IN, MAKES_MODEL, TRADES_AS, INVESTED_IN, COMPETES_WITH, "
    "ACQUIRED, PARTNERED_WITH, BASED_IN. "
    "Return ONLY JSON of the form {\"triples\": [[\"subject\",\"PREDICATE\",\"object\"], ...]}. "
    "Extract only SALIENT, factual relations; ignore opinions and filler. "
    "Use canonical entity names (e.g. 'General Motors' not 'the automaker'). Do not invent facts. "
    "If the text has no clear factual relation, return {\"triples\": []}."
)

EXTRACT_FEWSHOT = (
    'Text: "Founded in 2015, Nikola Corporation (Nasdaq: NKLA) is headquartered in Phoenix, '
    'Arizona. NIO\'s CEO, William Li, unveiled the ET7 sedan."\n'
    'JSON: {"triples": [["Nikola","FOUNDED_IN","2015"],'
    '["Nikola","TRADES_AS","NKLA"],["Nikola","HEADQUARTERED_IN","Phoenix"],'
    '["NIO","CEO_IS","William Li"],["NIO","MAKES_MODEL","ET7"]]}'
)


def _good_triple(s, p, o):
    """Reject junk triples (URLs, times, overly long spans, self-loops) so the
    graph stays clean despite small-model extraction noise."""
    if not (s and p and o):
        return False
    for x in (s, o):
        if len(x) > 45 or len(x) < 2:
            return False
        if re.search(r"https?://|www\.|@|\.com|webcast|\d{1,2}:\d{2}", x, re.IGNORECASE):
            return False
    if s.lower() == o.lower():
        return False
    if len(p) > 35:  # the model sometimes emits sentence-long predicates
        return False
    return True


def extract_triples(text):
    """LLM -> list of cleaned (subj, pred, obj) triples for one passage."""
    prompt = f"{EXTRACT_FEWSHOT}\n\nText:\n{text}\n\nJSON:"
    raw = ollama_generate(prompt, system=EXTRACT_SYSTEM, fmt="json", num_predict=640)
    cand = []
    try:
        obj = json.loads(raw)
        for t in obj.get("triples", []):
            if isinstance(t, (list, tuple)) and len(t) == 3:
                cand.append(tuple(str(x).strip() for x in t))
    except json.JSONDecodeError:
        for m in re.finditer(r'\[\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\]', raw):
            cand.append(tuple(g.strip() for g in m.groups()))
    triples = []
    for s, p, o in cand:
        if _good_triple(s, p, o):
            triples.append((s, p.upper().replace(" ", "_").strip("_"), o))
    return triples


# ---------------------------------------------------------------------------
# Step 2 -- Graph construction with deduplication (normalisation)
# ---------------------------------------------------------------------------
def canonical(name):
    """Normalise an EV-entity surface form so duplicates collapse to one node."""
    n = name.strip().strip('."\'').strip()
    n = re.sub(r"\b(Corporation|Corp\.?|Inc\.?|Ltd\.?|LLC|Co\.?|Group|Motors? Company)\b\.?$",
               "", n, flags=re.IGNORECASE).strip().rstrip(",")
    aliases = {
        "gm": "General Motors", "general motors": "General Motors",
        "general motors company": "General Motors",
        "vw": "Volkswagen", "volkswagen group": "Volkswagen",
        "ford motor": "Ford", "ford motor company": "Ford",
        "tesla inc": "Tesla", "tesla motors": "Tesla",
        "nio inc": "NIO", "byd company": "BYD", "byd auto": "BYD",
        "rivian automotive": "Rivian", "lucid motors": "Lucid", "lucid group": "Lucid",
        "nikola corporation": "Nikola", "xpeng motors": "XPeng", "xpeng inc": "XPeng",
        "vinfast auto": "VinFast", "polestar automotive": "Polestar",
        "berkshire hathaway inc": "Berkshire Hathaway",
        "u.s.": "United States", "us": "United States", "usa": "United States",
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
# capitalized words that are NOT entities (question/function words)
_LINK_STOP = {
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "the", "a", "an", "in", "on", "of", "is", "are", "does", "do", "did", "name",
    "us", "u.s", "u.s.", "ev", "evs", "i", "and", "or", "for", "with", "to",
    "company", "maker", "makers", "city", "country", "year", "stock", "ticker",
    "model", "trade", "trades", "market", "vehicle", "vehicles",
}


def _tokens(s):
    return [w for w in re.findall(r"[a-z0-9][a-z0-9.\-]+", str(s).lower())]


def _question_keys(question):
    """Significant entity tokens of a question = proper nouns (capitalized words)
    + numbers. Keying off case avoids matching common verbs like 'trade'/'maker'
    that otherwise link junk nodes and dilute the retrieved subgraph."""
    proper = re.findall(r"\b[A-Z][A-Za-z0-9.\-]*\b", question)
    nums = re.findall(r"\b\d[\d,\.]*\b", question)
    keys = set()
    for w in proper + nums:
        wl = w.lower().strip(".")
        if wl and wl not in _LINK_STOP and len(wl) >= 2:
            keys.add(wl)
    return keys


def link_entities(question, G):
    """Find graph nodes referenced by the question.

    A node matches if its full surface form is a substring of the question, OR if
    it shares a proper-noun / number token with the question. Token matching is
    what lets the question word 'Gothenburg' link the compound node
    'GOTHENBURG, SWEDEN'.
    """
    q = question.lower()
    qkeys = _question_keys(question)
    scored = []
    for node in G.nodes():
        nl = str(node).lower()
        if len(nl) >= 3 and re.search(r"\b" + re.escape(nl) + r"\b", q):
            scored.append((node, 100 + len(nl)))  # full-name match: strongest
            continue
        ntokens = {t for t in _tokens(node) if len(t) >= 2}
        shared = qkeys.intersection(ntokens)
        if shared:
            spec = sum(len(t) for t in shared) - 0.15 * G.degree(node)
            scored.append((node, spec))
    scored.sort(key=lambda x: -x[1])
    final = []
    for node, _ in scored:
        if len(final) >= 5:
            break
        if not any(node != o and str(node).lower() in str(o).lower() for o in final):
            final.append(node)
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
