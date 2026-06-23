"""
run_indexing.py
===============
One-off (slow) indexing pass over the EV corpus:

    load 70 docs -> clean -> stride-sample chunks -> LLM triple extraction
    -> NetworkX graph -> save artifacts/

Run once; the notebook then loads the cached artifacts so it executes fast and
reproducibly. Re-run only if you change the corpus or extraction prompt.
"""
import json
import os
import time

import networkx as nx

import graphrag_core as gc
from corpus_loader import load_documents, build_chunks

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ART, exist_ok=True)

SAMPLE = dict(size=900, overlap=120, per_doc=5, stride=True)

CKPT = os.path.join(ART, "_checkpoint.json")


def main():
    docs = load_documents()
    chunks = build_chunks(docs, **SAMPLE)
    print(f"Loaded {len(docs)} docs -> {len(chunks)} sampled chunks", flush=True)

    # ---- resume from checkpoint if it matches this chunk set ----
    start, all_triples = 0, []
    gc.reset_stats()
    elapsed0 = 0.0
    if os.path.exists(CKPT):
        ck = json.load(open(CKPT, encoding="utf-8"))
        if ck.get("n_chunks") == len(chunks):
            start = ck["next_index"]
            all_triples = [tuple(t) for t in ck["triples"]]
            for k in gc.STATS:
                gc.STATS[k] = ck["stats"][k]
            elapsed0 = ck["seconds"]
            print(f"Resuming from checkpoint at chunk {start} "
                  f"({len(all_triples)} triples so far)", flush=True)

    t0 = time.time()
    for i in range(start, len(chunks)):
        all_triples.extend(gc.extract_triples(chunks[i]["text"]))
        if (i + 1) % 10 == 0 or i == 0:
            el = elapsed0 + time.time() - t0
            print(f"[{i+1:3d}/{len(chunks)}] triples so far={len(all_triples)} "
                  f"| {el:.0f}s | {el/(i+1):.1f}s/chunk", flush=True)
        if (i + 1) % 20 == 0:  # checkpoint
            json.dump({"n_chunks": len(chunks), "next_index": i + 1,
                       "triples": all_triples, "stats": dict(gc.STATS),
                       "seconds": elapsed0 + time.time() - t0},
                      open(CKPT, "w", encoding="utf-8"), ensure_ascii=False)

    index_seconds = elapsed0 + time.time() - t0
    index_stats = dict(gc.STATS)
    G = gc.build_graph(all_triples)
    print(f"\nDONE: {len(all_triples)} triples in {index_seconds:.0f}s | "
          f"graph {G.number_of_nodes()} nodes / {G.number_of_edges()} edges", flush=True)

    # ---- save artifacts ----
    with open(os.path.join(ART, "triples.json"), "w", encoding="utf-8") as f:
        json.dump(all_triples, f, ensure_ascii=False)
    with open(os.path.join(ART, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(os.path.join(ART, "index_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": index_stats, "seconds": index_seconds,
                   "n_chunks": len(chunks), "n_docs": len(docs),
                   "n_triples": len(all_triples),
                   "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                   "sample": SAMPLE, "model": gc.LLM_MODEL}, f, ensure_ascii=False, indent=2)
    nx.write_graphml(G, os.path.join(ART, "graph.graphml"))
    print("Saved artifacts/ (triples.json, chunks.json, index_stats.json, graph.graphml)", flush=True)


if __name__ == "__main__":
    main()
