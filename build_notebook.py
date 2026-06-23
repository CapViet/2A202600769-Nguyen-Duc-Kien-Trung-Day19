# -*- coding: utf-8 -*-
"""Assemble the Day-19 GraphRAG deliverable notebook for the lecturer's
US Electric Vehicle corpus. The heavy indexing is done once by run_indexing.py;
this notebook loads cached artifacts/ so it executes fast and reproducibly.
"""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

# ---------------------------------------------------------------- Title
md(r"""# LAB DAY 19 — Xây dựng hệ thống **GraphRAG** với *US Electric Vehicle Corpus*

**Sinh viên:** Nguyễn Đức Kiên Trung &nbsp;•&nbsp; **MSSV:** 2A202600769 &nbsp;•&nbsp; **Ngày:** 2026-06-23

Pipeline **GraphRAG** hoàn chỉnh trên **bộ dữ liệu thật do giảng viên cung cấp** (`dataset/` — 70 tài liệu
web về ngành **xe điện (EV) Mỹ**), so sánh với **Flat RAG**. Toàn bộ chạy **offline / miễn phí**:

| Thành phần | Công cụ |
|---|---|
| LLM (trích xuất + sinh câu trả lời) | **Ollama** · `qwen2.5:3b` |
| Embeddings (Flat RAG) | `sentence-transformers · all-MiniLM-L6-v2` |
| Vector index (Flat RAG) | **FAISS** (cosine) |
| Đồ thị tri thức (GraphRAG) | **NetworkX** (MultiDiGraph) |
| Trực quan hóa | **Matplotlib** |

### Bộ dữ liệu
70 file `doc_1.txt … doc_70.txt`, mỗi file có cấu trúc `Query / Title / Link / Snippet / Full Content`,
thuộc 8 chủ đề truy vấn về EV Mỹ (sentiment, financial performance, investor sentiment, market trends,
regional analysis, US-vs-global, top EV stocks).

### Quy trình
1. **Load + clean** 70 tài liệu (lọc boilerplate web + nhiễu nhị phân từ file PDF).
2. **Indexing:** LLM trích xuất **triples** `(subject, PREDICATE, object)` → dựng **Knowledge Graph** (khử trùng lặp).
3. **Querying:** truy vấn **đa bước (2-hop)** bằng entity-linking → BFS → textualization → LLM.
4. **Evaluation:** so sánh **Flat RAG vs GraphRAG** trên **20 câu hỏi benchmark** + phân tích chi phí.

> ⚙️ **Chi phí & quy mô:** Bước Indexing (gọi LLM cho từng chunk) là phần tốn kém nhất nên được chạy **một lần**
> bởi `run_indexing.py` và lưu vào `artifacts/`. Notebook này **nạp lại artifacts** để chạy nhanh, tái lập được.
> Vì lý do chi phí, đồ thị được dựng từ **mẫu chunk trải đều** mỗi tài liệu (không phải toàn bộ 70 file) —
> đây là một đánh đổi cost/coverage điển hình của GraphRAG ở quy mô lớn.
""")

# ---------------------------------------------------------------- Part 1 research
md(r"""## Phần 1 — Nghiên cứu (Research)

**1. Entity Extraction — LLM phân biệt thực thể (Node) và thuộc tính (Property) thế nào?**
Thực thể là đối tượng có danh tính riêng, được tham chiếu lại và làm chủ/tân ngữ của nhiều quan hệ
(hãng EV như *Tesla, NIO, Polestar*; người; mẫu xe; mã cổ phiếu) → **Node**. Thuộc tính là giá trị mô tả
gắn với một thực thể (năm thành lập, thành phố, mã ticker) → biểu diễn bằng **cạnh tới một literal**.
Ở đây mọi thứ là triple `(subject, PREDICATE, object)`; vd `(Nikola, FOUNDED_IN, 2015)`.

**2. Graph Construction — Vì sao khử trùng lặp (Deduplication) quan trọng?**
LLM sinh nhiều biến thể cho cùng thực thể: *"GM" / "General Motors" / "the automaker"*,
*"VW" / "Volkswagen Group"*. Không hợp nhất → đồ thị **phân mảnh**, chuỗi suy luận đa bước **đứt gãy**
và số node phình vô nghĩa. `canonical()` chuẩn hóa tên + gộp alias để **một thực thể = một node**.

**3. Query Answering — Khác biệt BFS trên đồ thị vs tìm kiếm vector?**
*Vector search (Flat RAG)* trả về các đoạn **tương tự bề mặt** với câu hỏi, **không hiểu liên kết** giữa các
sự kiện → với câu hỏi đa bước (vd *"mã cổ phiếu của hãng EV đặt trụ sở ở Gothenburg là gì?"*) nó dễ lấy nhầm
đoạn và **bịa**. *BFS trên đồ thị (GraphRAG)* đi theo **quan hệ tường minh**:
`Gothenburg ← HEADQUARTERED_IN ← Polestar → TRADES_AS → PSNY`, lần theo **cấu trúc tri thức** thay vì độ tương tự.
""")

# ---------------------------------------------------------------- Setup
md(r"""## Phần 2 — Setup & Load corpus

```bash
pip install networkx matplotlib pandas faiss-cpu sentence-transformers nbformat jupyter tabulate
# LLM cục bộ: https://ollama.com  ->  ollama pull qwen2.5:3b
```
`USE_TF=0` được set trước khi import `sentence-transformers` để tránh xung đột TensorFlow/Keras 3.""")

code('import os\nos.environ["USE_TF"] = "0"; os.environ["USE_TORCH"] = "1"\n'
     'os.environ["TOKENIZERS_PARALLELISM"] = "false"\n'
     'import json, time, re\nimport networkx as nx\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n'
     'import graphrag_core as gc\nfrom corpus_loader import load_documents, build_chunks\n'
     'from benchmark_ev import BENCHMARK\nprint("Imports OK | LLM =", gc.LLM_MODEL)')

code('docs = load_documents()\n'
     'print(f"Đã nạp {len(docs)} tài liệu\\n")\n'
     'from collections import Counter\n'
     'themes = Counter(d["query"] for d in docs)\n'
     'print("8 chủ đề truy vấn:")\n'
     'for q, n in themes.items():\n    print(f"  • ({n:2d} docs) {q}")\n'
     'print("\\nVí dụ doc_1:")\n'
     'print("  Title :", docs[0]["title"])\n'
     'print("  Body  :", docs[0]["body"][:240], "...")')

# ---------------------------------------------------------------- Step 1 indexing
md(r"""## Bước 1 — Trích xuất Thực thể & Quan hệ (Indexing)

Mỗi tài liệu được **làm sạch** và **chia chunk** (~900 ký tự). Vì gọi LLM cho mọi chunk rất tốn kém, ta lấy
**mẫu trải đều 5 chunk/tài liệu** (`stride sampling`) — bắt được các sự kiện nằm rải khắp bài viết dài.
LLM đọc từng chunk và trả JSON các triple; bộ lọc `_good_triple()` loại bỏ rác (URL, giờ giấc, span quá dài).

> Phần này đã chạy sẵn bởi `run_indexing.py`; ở đây ta **nạp lại** kết quả từ `artifacts/`.""")

code(r'''ART = "artifacts"
ALL_TRIPLES = [tuple(t) for t in json.load(open(f"{ART}/triples.json", encoding="utf-8"))]
CHUNKS = json.load(open(f"{ART}/chunks.json", encoding="utf-8"))
META = json.load(open(f"{ART}/index_stats.json", encoding="utf-8"))
INDEX_STATS, INDEX_SECONDS = META["stats"], META["seconds"]
print(f"Sampled chunks   : {META['n_chunks']} (từ {META['n_docs']} docs)")
print(f"Triples trích xuất: {len(ALL_TRIPLES)}")
print(f"Thời gian indexing: {INDEX_SECONDS:.0f}s | tokens={INDEX_STATS}")
print("\\nVí dụ 12 triple:")
for s, p, o in ALL_TRIPLES[:12]:
    print(f"   ({s})  --{p}-->  ({o})")''')

md("**Demo trực tiếp** — chạy lại bộ trích xuất trên 1 chunk để minh hoạ cơ chế:")
code(r'''demo_chunk = next(c for c in CHUNKS if "Nikola" in c["text"] or "Polestar" in c["text"])
print("Chunk nguồn:", demo_chunk["text"][:260], "...\n")
print("Triple LLM trích ra:")
for t in gc.extract_triples(demo_chunk["text"]):
    print("   ", t)''')

# ---------------------------------------------------------------- Step 2 construction
md(r"""## Bước 2 — Xây dựng Đồ thị (Construction) + Khử trùng lặp

`build_graph()` chuẩn hóa tên qua `canonical()` (gộp *GM ↔ General Motors*, *VW ↔ Volkswagen*…) rồi thêm
node/edge, loại cạnh trùng. Đồ thị thật khá lớn nên ta xem các **hub** (node nhiều liên kết nhất).""")

code(r'''G = gc.build_graph(ALL_TRIPLES)
print(f"Đồ thị tri thức: {G.number_of_nodes()} node, {G.number_of_edges()} cạnh")
print("\\nTop-15 hub (degree cao nhất):")
for n, d in sorted(G.degree(), key=lambda x: -x[1])[:15]:
    print(f"   {str(n)[:28]:28s} degree={d}")''')

md(r"""### Trực quan hóa (Deliverable #2)
Đồ thị đầy đủ quá dày để hiển thị; ta vẽ **subgraph của ~45 node có degree cao nhất** và lưu
`knowledge_graph.png`.""")
code(r'''deg = dict(G.degree())
top_nodes = [n for n, _ in sorted(deg.items(), key=lambda x: -x[1])[:45]]
H = G.subgraph(top_nodes)

plt.figure(figsize=(22, 15))
pos = nx.spring_layout(H, k=1.1, iterations=90, seed=42)
sizes = [300 + 220 * deg[n] for n in H.nodes()]
nx.draw_networkx_nodes(H, pos, node_color="#ff7043", node_size=sizes, alpha=0.9)
nx.draw_networkx_edges(H, pos, edge_color="#90a4ae", alpha=0.45, arrows=True,
                       arrowsize=10, connectionstyle="arc3,rad=0.08")
nx.draw_networkx_labels(H, pos, font_size=8, font_weight="bold")
elabels = {(u, v): d["relation"] for u, v, d in H.edges(data=True)}
nx.draw_networkx_edge_labels(H, pos, edge_labels=elabels, font_size=5, font_color="#607d8b")
plt.title("US EV Knowledge Graph — top-45 hub nodes (NetworkX)", fontsize=15)
plt.axis("off"); plt.tight_layout()
plt.savefig("knowledge_graph.png", dpi=130, bbox_inches="tight")
print("Đã lưu knowledge_graph.png")
plt.show()''')

# ---------------------------------------------------------------- Step 3 query
md(r"""## Bước 3 — Truy vấn GraphRAG (Multi-hop)

**(1)** câu hỏi → **(2)** entity-linking tìm node trong câu → **(3)** BFS 2-hop lấy subgraph lân cận →
**(4)** textualization → LLM sinh câu trả lời chỉ dựa trên facts của đồ thị.""")
code(r'''demo_q = "What stock ticker does Polestar trade under?"
ans, ctx, seeds = gc.graphrag_answer(demo_q, G, k=2)
print("Câu hỏi :", demo_q)
print("Seeds   :", seeds)
print("Subgraph facts:")
for line in ctx.split("\n")[:12]:
    print("   ", line)
print("\nGraphRAG:", ans)''')

# ---------------------------------------------------------------- Flat RAG
md(r"""## Baseline — Flat RAG (FAISS + MiniLM)

Để so sánh **công bằng**, Flat RAG index **đúng tập chunk** mà GraphRAG đã dùng để dựng đồ thị
(cùng cơ sở tri thức; biến số duy nhất là **cách truy hồi**). Mỗi chunk được embed bằng MiniLM, đánh chỉ mục
FAISS, truy hồi top-k đoạn tương tự nhất rồi đưa cho cùng một LLM.""")
code(r'''flat = gc.FlatRAG([c["text"] for c in CHUNKS])
a, passages = flat.answer(demo_q, k=4)
print("Flat RAG top-4 passages (rút gọn):")
for p in passages:
    print("   -", p[:110], "...")
print("\nFlat RAG:", a)''')

# ---------------------------------------------------------------- Step 4 eval
md(r"""## Bước 4 — So sánh & Đánh giá (Evaluation)

**20 câu hỏi benchmark** (tự soạn, có đáp án kiểm chứng được từ corpus; trộn single-hop & multi-hop) chạy trên
cả hai hệ thống. Một câu **đúng** nếu mọi token định danh cốt lõi của đáp án tham chiếu có trong câu trả lời
(bỏ phần trong ngoặc — chỉ là chú thích) và câu trả lời không phải *"I don't know"*.""")
code(r'''def is_correct(answer, ref):
    a = answer.lower()
    if "i don't know" in a or "i do not know" in a:
        return False
    core = re.sub(r"\(.*?\)", " ", ref)
    keys = [w for w in re.findall(r"[A-Za-z0-9.\-]+", core) if w[0].isupper() or w[0].isdigit()]
    stop = {"founder","of","ceo","the","and","in","name","a","model","city","country","maker","company"}
    keys = [w.lower() for w in keys if w.lower() not in stop and len(w) > 1]
    return all(k in a for k in keys) if keys else False

reset = gc.reset_stats(); gc.reset_stats()
rows = []
for item in BENCHMARK:
    q, ref, hop = item["q"], item["ref"], item["hop"]
    g_ans, _, _ = gc.graphrag_answer(q, G, k=2)
    f_ans, _ = flat.answer(q, k=4)
    rows.append({"hop": hop, "question": q, "reference": ref,
                 "flat_answer": f_ans, "flat_ok": is_correct(f_ans, ref),
                 "graph_answer": g_ans, "graph_ok": is_correct(g_ans, ref)})
    print(f"[hop {hop}] {q[:44]:44s} | Flat {'OK ' if rows[-1]['flat_ok'] else 'X  '}"
          f"| Graph {'OK' if rows[-1]['graph_ok'] else 'X'}")
EVAL_STATS = dict(gc.STATS)
df = pd.DataFrame(rows)
print("\nXong 20 câu hỏi.")''')

md("### Bảng so sánh 20 câu hỏi (Deliverable #3)")
code('pd.set_option("display.max_colwidth", 55)\n'
     'df[["hop","question","flat_ok","graph_ok","flat_answer","graph_answer"]]')

md("### Tổng hợp độ chính xác")
code(r'''multi = df[df["hop"] >= 2]
summary = pd.DataFrame({
    "Hệ thống": ["Flat RAG", "GraphRAG"],
    "Đúng / 20": [int(df["flat_ok"].sum()), int(df["graph_ok"].sum())],
    "Accuracy (tất cả)": [f"{df['flat_ok'].mean():.0%}", f"{df['graph_ok'].mean():.0%}"],
    "Accuracy (đa bước)": [f"{multi['flat_ok'].mean():.0%}", f"{multi['graph_ok'].mean():.0%}"],
})
summary''')

md(r"""### Các trường hợp Flat RAG **bị ảo giác / sai** nhưng GraphRAG **đúng**""")
code(r'''halluc = df[(~df["flat_ok"]) & (df["graph_ok"])]
print(f"Có {len(halluc)} trường hợp Flat RAG sai/ảo giác mà GraphRAG đúng:\n")
for _, r in halluc.iterrows():
    print("Q :", r["question"])
    print("  Flat RAG  (sai) :", r["flat_answer"])
    print("  GraphRAG  (đúng):", r["graph_answer"])
    print("  Đáp án          :", r["reference"], "\n")''')

# ---------------------------------------------------------------- Cost
md(r"""## Phân tích chi phí (Deliverable #4) — Token usage & Time

Hai giai đoạn: **Indexing** (dựng đồ thị, chạy 1 lần, tốn nhất) và **Evaluation** (truy vấn). Dùng Ollama cục bộ
nên **chi phí tiền tệ = $0.00**; ta đo token + thời gian làm đại lượng thay thế.""")
code(r'''cost = pd.DataFrame([
    {"Giai đoạn": f"Indexing ({META['n_chunks']} chunks → triples)",
     "LLM calls": INDEX_STATS["calls"], "Prompt tokens": INDEX_STATS["prompt_tokens"],
     "Completion tokens": INDEX_STATS["completion_tokens"],
     "Tổng tokens": INDEX_STATS["prompt_tokens"] + INDEX_STATS["completion_tokens"],
     "Thời gian (s)": round(INDEX_SECONDS, 1)},
    {"Giai đoạn": "Evaluation (20 Q × 2 hệ thống)",
     "LLM calls": EVAL_STATS["calls"], "Prompt tokens": EVAL_STATS["prompt_tokens"],
     "Completion tokens": EVAL_STATS["completion_tokens"],
     "Tổng tokens": EVAL_STATS["prompt_tokens"] + EVAL_STATS["completion_tokens"],
     "Thời gian (s)": round(EVAL_STATS["seconds"], 1)},
])
avg = INDEX_SECONDS / max(META["n_chunks"], 1)
print(f"Mô hình: {gc.LLM_MODEL} | Chi phí tiền tệ: $0.00 (Ollama cục bộ)")
print(f"Tốc độ trích xuất TB: {avg:.1f}s/chunk | "
      f"Ngoại suy toàn bộ ~2000 chunk ≈ {avg*2000/60:.0f} phút (lý do phải lấy mẫu)")
cost''')

# ---------------------------------------------------------------- Persist
md(r"""## Lưu kết quả ra file (để output không bị mất)

Ghi mọi bảng xuống đĩa (`results_20q_comparison.csv`, `cost_analysis.csv`, `RESULTS.md`) — dù notebook bị
**Clear Outputs** hay chạy lại, dữ liệu vẫn còn.""")
code(r'''df.to_csv("results_20q_comparison.csv", index=False, encoding="utf-8-sig")
cost.to_csv("cost_analysis.csv", index=False, encoding="utf-8-sig")
lines = ["# KẾT QUẢ LAB DAY 19 — GraphRAG vs Flat RAG (US EV Corpus)\n",
         f"- LLM: `{gc.LLM_MODEL}` (Ollama, local) · Chi phí tiền tệ: **$0.00**",
         f"- Corpus: {META['n_docs']} docs → {META['n_chunks']} sampled chunks · "
         f"Graph: NetworkX ({G.number_of_nodes()} node, {G.number_of_edges()} cạnh)\n",
         "## Tổng hợp độ chính xác", summary.to_markdown(index=False),
         "\n## Deliverable #4 — Chi phí (Token usage & Time)", cost.to_markdown(index=False),
         "\n## Deliverable #3 — Bảng so sánh 20 câu hỏi",
         df[["hop","question","flat_ok","graph_ok","flat_answer","graph_answer"]].to_markdown(index=False),
         "\n## Các trường hợp Flat RAG sai nhưng GraphRAG đúng"]
for _, r in halluc.iterrows():
    lines += [f"\n**Q:** {r['question']}", f"- Flat RAG (sai): {r['flat_answer']}",
              f"- GraphRAG (đúng): {r['graph_answer']}", f"- Đáp án: {r['reference']}"]
open("RESULTS.md", "w", encoding="utf-8").write("\n".join(lines))
print("Đã lưu: results_20q_comparison.csv, cost_analysis.csv, RESULTS.md")''')

# ---------------------------------------------------------------- Conclusion
md(r"""## Kết luận

- **GraphRAG mạnh ở câu hỏi đa bước:** duyệt cạnh quan hệ tường minh (BFS 2-hop) giúp ghép các sự kiện rời
  rạc (vd *trụ sở → hãng → mã cổ phiếu*) mà Flat RAG — chỉ dựa độ tương tự vector — bỏ sót và hay **bịa**.
- **Khử trùng lặp** giữ cho chuỗi suy luận không đứt gãy khi cùng một hãng xuất hiện dưới nhiều tên.
- **Đánh đổi chi phí (dữ liệu thật):** Indexing tốn nhiều LLM-call và là nút cổ chai → phải **lấy mẫu chunk**;
  trên corpus thật, **chất lượng trích xuất của mô hình 3B là yếu tố giới hạn** chính của GraphRAG.
  Flat RAG index rẻ (chỉ embed) và truy hồi trực tiếp tốt cho câu single-hop.
- Hướng cải thiện: mô hình trích xuất mạnh hơn, chuẩn hóa thực thể tốt hơn, và chuyển sang **Neo4j**
  (Cypher + Bloom) khi mở rộng quy mô.

**Deliverables:** ✅ mã nguồn (notebook + modules) · ✅ ảnh đồ thị `knowledge_graph.png` ·
✅ bảng so sánh 20 câu hỏi (`RESULTS.md`, CSV) · ✅ phân tích token/time.
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
with open("Day19_GraphRAG.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote Day19_GraphRAG.ipynb with {len(cells)} cells")
