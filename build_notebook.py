# -*- coding: utf-8 -*-
"""Assemble the self-contained Day-19 GraphRAG deliverable notebook.

The core library (graphrag_core.py) and corpus (corpus.py) are inlined verbatim
into notebook cells so the .ipynb stands alone for submission, while remaining
identical to the code already validated against Ollama.
"""
import re
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def strip_module_docstring(src: str) -> str:
    """Remove the leading triple-quoted module docstring."""
    m = re.match(r'\s*(?:#[^\n]*\n)*\s*(?:"""|\'\'\')', src)
    if m:
        q = src[m.end() - 3:m.end()]
        end = src.index(q, m.end())
        return src[end + 3:].lstrip("\n")
    return src


with open("corpus.py", encoding="utf-8") as f:
    corpus_src = strip_module_docstring(f.read())
with open("graphrag_core.py", encoding="utf-8") as f:
    core_src = strip_module_docstring(f.read())

cells = []
md = lambda s: cells.append(new_markdown_cell(s))
code = lambda s: cells.append(new_code_cell(s))

# ---------------------------------------------------------------- Title
md(r"""# LAB DAY 19 — Xây dựng hệ thống **GraphRAG** với Tech Company Corpus

**Sinh viên:** Nguyễn Đức Kiên Trung &nbsp;•&nbsp; **MSSV:** 2A202600769 &nbsp;•&nbsp; **Ngày:** 2026-06-23

Notebook này xây dựng một pipeline **GraphRAG** hoàn chỉnh và so sánh với **Flat RAG**.
Toàn bộ chạy **offline / miễn phí** bằng stack cục bộ:

| Thành phần | Công cụ |
|---|---|
| LLM (trích xuất + sinh câu trả lời) | **Ollama** · `qwen2.5:3b` |
| Embeddings (Flat RAG) | `sentence-transformers · all-MiniLM-L6-v2` |
| Vector DB (Flat RAG) | **FAISS** |
| Đồ thị tri thức (GraphRAG) | **NetworkX** (MultiDiGraph) |
| Trực quan hóa | **Matplotlib** |

### Mục tiêu
1. Trích xuất thực thể (Entity) & quan hệ (Relation) từ văn bản thô → **Triples**.
2. Dựng **Knowledge Graph** bằng NetworkX (có khử trùng lặp).
3. Truy vấn **đa bước (2-hop)**: entity linking → BFS → textualization → LLM.
4. So sánh độ chính xác **Flat RAG vs GraphRAG** trên 20 câu hỏi benchmark + phân tích chi phí.
""")

# ---------------------------------------------------------------- Part 1 research
md(r"""## Phần 1 — Nghiên cứu (Research)

**1. Entity Extraction — LLM phân biệt thực thể (Node) và thuộc tính (Property) thế nào?**
LLM dựa vào vai trò ngữ nghĩa trong câu. *Thực thể* là đối tượng có danh tính riêng, có thể được
tham chiếu lại và đứng làm chủ/tân ngữ của nhiều quan hệ (công ty, người, sản phẩm) → trở thành **Node**.
*Thuộc tính* là giá trị mô tả gắn liền một thực thể, thường không có quan hệ riêng (năm thành lập, thành phố) →
gắn vào node dưới dạng **edge tới một literal** hoặc property. Trong lab này ta mô hình hoá mọi thứ thành
triple `(subject, PREDICATE, object)`; những giá trị như năm `2015` là object literal của quan hệ `FOUNDED_IN`.

**2. Graph Construction — Vì sao khử trùng lặp (Deduplication) lại quan trọng?**
LLM tạo ra nhiều biến thể bề mặt cho cùng một thực thể: *"Google"*, *"Google LLC"*, *"the company"*.
Nếu không hợp nhất, đồ thị bị **phân mảnh**: các sự kiện về cùng một thực thể nằm rải ở nhiều node khác nhau,
khiến việc duyệt đa bước **đứt gãy** (không tìm được đường đi) và làm phình số node vô nghĩa.
Khử trùng lặp (chuẩn hóa tên + gộp alias) đảm bảo **một thực thể = một node**, giữ cho các chuỗi quan hệ liền mạch.

**3. Query Answering — Khác biệt giữa duyệt đồ thị (BFS) và tìm kiếm vector?**
*Vector search (Flat RAG)* trả về các đoạn văn **tương tự bề mặt** với câu hỏi; nó **không có khái niệm liên kết**
giữa các sự kiện, nên với câu hỏi đa bước ("Ai sáng lập công ty mà Google mua năm 2014?") nó thường lấy nhầm đoạn
và **bịa (hallucinate)**. *BFS trên đồ thị (GraphRAG)* đi theo **quan hệ tường minh**: từ node `Google` → cạnh
`ACQUIRED` → `DeepMind` → cạnh `FOUNDED_BY` → `Demis Hassabis`. Nó **lần theo cấu trúc tri thức** thay vì độ tương tự,
nên giải được suy luận nhiều bước.
""")

# ---------------------------------------------------------------- Part 2 setup
md(r"""## Phần 2 — Environment Setup

Cài đặt (chỉ chạy 1 lần). Lab dùng Ollama cục bộ thay cho OpenAI để **miễn phí & offline**:

```bash
pip install networkx matplotlib neo4j openai pandas
pip install langchain langchain-openai faiss-cpu sentence-transformers
# LLM cục bộ:  https://ollama.com  ->  ollama pull qwen2.5:3b
```
> `USE_TF=0` được set trước khi import `sentence-transformers` để tránh xung đột TensorFlow/Keras-3.
""")

code('import os\nos.environ["USE_TF"] = "0"\nos.environ["USE_TORCH"] = "1"\nos.environ["TOKENIZERS_PARALLELISM"] = "false"\n\n'
     'import json, re, time, urllib.request\nfrom collections import defaultdict\n'
     'import networkx as nx\nimport matplotlib.pyplot as plt\nimport pandas as pd\n'
     'print("Imports OK")')

# ---------------------------------------------------------------- Corpus cell
md(r"""## Bộ dữ liệu — *Tech Company Corpus*

42 câu sự kiện về các công ty công nghệ. Nhiều sự kiện được **móc xích** qua nhiều câu
(vd: *DeepMind → sáng lập bởi Demis Hassabis*; *DeepMind → bị Google mua lại*) — đây chính là phần
GraphRAG duyệt được còn Flat RAG hay bỏ sót.""")
code(corpus_src)
code('print(f"Corpus: {len(CORPUS)} câu | Benchmark: {len(BENCHMARK)} câu hỏi")\n'
     'for d in CORPUS[:5]:\n    print(" •", d)')

# ---------------------------------------------------------------- Core library cell
md(r"""## Thư viện lõi GraphRAG

Toàn bộ logic (gọi Ollama + đếm token, trích xuất triple, dựng đồ thị có khử trùng lặp,
truy vấn 2-hop, và baseline Flat RAG) gói trong một cell để notebook **độc lập**.""")
code(core_src)

# ---------------------------------------------------------------- Step 1 indexing
md(r"""## Bước 1 — Trích xuất Thực thể & Quan hệ (Indexing)

Dùng LLM đọc từng câu và trả về JSON các triple `(subject, PREDICATE, object)`. Few-shot + ràng buộc
`format="json"` của Ollama giúp output ổn định, dễ parse.""")
code('reset_stats()\nALL_TRIPLES = []\nt0 = time.time()\n'
     'for i, doc in enumerate(CORPUS):\n'
     '    triples = extract_triples(doc)\n'
     '    ALL_TRIPLES.extend(triples)\n'
     '    print(f"[{i+1:2d}/{len(CORPUS)}] {len(triples)} triple  | {doc[:48]}")\n'
     'INDEX_STATS = dict(STATS)\nINDEX_SECONDS = time.time() - t0\n'
     'print(f"\\n==> {len(ALL_TRIPLES)} triple trong {INDEX_SECONDS:.1f}s | tokens={INDEX_STATS}")')
code('print("Ví dụ 12 triple đầu tiên:")\n'
     'for s, p, o in ALL_TRIPLES[:12]:\n    print(f"   ({s})  --{p}-->  ({o})")')

# ---------------------------------------------------------------- Step 2 construction
md(r"""## Bước 2 — Xây dựng Đồ thị (Construction) + Khử trùng lặp

`canonical()` chuẩn hóa tên (gộp *Google LLC → Google*, *Facebook → Meta*…) trước khi thêm node,
đảm bảo **một thực thể = một node**. Cạnh trùng lặp `(subj, pred, obj)` bị loại bỏ.""")
code('G = build_graph(ALL_TRIPLES)\n'
     'print(f"Đồ thị: {G.number_of_nodes()} node, {G.number_of_edges()} cạnh")\n'
     'print("Bậc cao nhất (hub):")\n'
     'for n, d in sorted(G.degree(), key=lambda x: -x[1])[:8]:\n    print(f"   {n:14s} degree={d}")')

md(r"""### Trực quan hóa đồ thị tri thức (Deliverable #2)
Lưu ra `knowledge_graph.png` để nộp kèm báo cáo.""")
code(r'''plt.figure(figsize=(20, 14))
pos = nx.spring_layout(G, k=0.9, iterations=80, seed=42)

# màu node: công ty (hub, degree cao) vs thực thể khác
deg = dict(G.degree())
companies = {"OpenAI","Google","Alphabet","DeepMind","Microsoft","Meta","Nvidia",
             "Tesla","SpaceX","Anthropic","Apple","YouTube","Instagram","WhatsApp","GitHub"}
node_colors = ["#ff7043" if n in companies else "#4fc3f7" for n in G.nodes()]
node_sizes  = [600 + 300 * deg[n] for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.95)
nx.draw_networkx_edges(G, pos, edge_color="#9e9e9e", alpha=0.5, arrows=True,
                       arrowsize=12, connectionstyle="arc3,rad=0.08")
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
edge_labels = {(u, v): d["relation"] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_color="#616161")
plt.title("Tech Company Knowledge Graph (NetworkX) — cam = công ty, xanh = thực thể/literal",
          fontsize=15)
plt.axis("off"); plt.tight_layout()
plt.savefig("knowledge_graph.png", dpi=130, bbox_inches="tight")
print("Đã lưu knowledge_graph.png")
plt.show()''')

# ---------------------------------------------------------------- Step 3 querying
md(r"""## Bước 3 — Truy vấn GraphRAG (Multi-hop)

Quy trình: **(1)** lấy câu hỏi → **(2)** entity linking tìm node trong câu → **(3)** BFS 2-hop lấy
subgraph lân cận → **(4)** textualization thành đoạn văn → gửi LLM sinh câu trả lời.""")
code(r'''demo_q = "Who founded the company that Google acquired in 2014?"
ans, ctx, seeds = graphrag_answer(demo_q, G, k=2)
print("Câu hỏi :", demo_q)
print("Seeds   :", seeds)
print("Subgraph context (textualized):")
for line in ctx.split("\n"):
    print("   ", line)
print("\nGraphRAG trả lời:", ans)''')

# ---------------------------------------------------------------- Flat RAG
md(r"""## Baseline — Flat RAG (FAISS + MiniLM)

Mỗi câu trong corpus là một passage; embed bằng MiniLM, đánh chỉ mục bằng FAISS (cosine similarity),
truy hồi top-k passage tương tự nhất rồi đưa cho cùng một LLM. Đây là baseline để so sánh.""")
code('flat = FlatRAG(CORPUS)\n'
     'a, ctx = flat.answer(demo_q, k=4)\n'
     'print("Flat RAG top-4 passages:")\n'
     'for c in ctx:\n    print("   -", c)\n'
     'print("\\nFlat RAG trả lời:", a)')

# ---------------------------------------------------------------- Step 4 evaluation
md(r"""## Bước 4 — So sánh & Đánh giá (Evaluation)

Chạy **20 câu hỏi benchmark** trên cả hai hệ thống. Một câu được tính **đúng** nếu mọi token cốt lõi của
đáp án tham chiếu xuất hiện trong câu trả lời (và không phải *"I don't know"*). Cột `hop` cho biết câu hỏi
cần ghép ≥2 sự kiện (đa bước).""")
code(r'''def is_correct(answer, ref):
    """Đúng nếu MỌI token định danh cốt lõi của đáp án tham chiếu có trong câu trả lời.
    Bỏ phần giải thích trong ngoặc đơn (vd '(founder of DeepMind)') vì đó là chú thích,
    không phải token bắt buộc."""
    a = answer.lower()
    if "i don't know" in a or "i do not know" in a:
        return False
    core = re.sub(r"\(.*?\)", " ", ref)  # bỏ phần trong ngoặc
    keys = [w for w in re.findall(r"[A-Za-z0-9]+", core) if w[0].isupper() or w.isdigit()]
    stop = {"founder","of","ceo","the","and","invested","in","name","a","model"}
    keys = [w.lower() for w in keys if w.lower() not in stop]
    return all(k in a for k in keys) if keys else False

reset_stats(); rows = []
for item in BENCHMARK:
    q, ref, hop = item["q"], item["ref"], item["hop"]
    g_ans, _, _ = graphrag_answer(q, G, k=2)
    f_ans, _ = flat.answer(q, k=4)
    rows.append({
        "hop": hop, "question": q, "reference": ref,
        "flat_answer": f_ans, "flat_ok": is_correct(f_ans, ref),
        "graph_answer": g_ans, "graph_ok": is_correct(g_ans, ref),
    })
    print(f"[hop {hop}] {q[:46]:46s} | Flat {'OK ' if rows[-1]['flat_ok'] else 'X  '}"
          f"| Graph {'OK' if rows[-1]['graph_ok'] else 'X'}")
EVAL_STATS = dict(STATS)
df = pd.DataFrame(rows)
print("\nĐã chạy xong 20 câu hỏi.")''')

md("### Bảng so sánh 20 câu hỏi (Deliverable #3)")
code('pd.set_option("display.max_colwidth", 60)\n'
     'df[["hop","question","flat_ok","graph_ok","flat_answer","graph_answer"]]')

md("### Tổng hợp độ chính xác")
code(r'''flat_acc  = df["flat_ok"].mean()
graph_acc = df["graph_ok"].mean()
multi = df[df["hop"] >= 2]
summary = pd.DataFrame({
    "Hệ thống": ["Flat RAG", "GraphRAG"],
    "Đúng / 20": [df["flat_ok"].sum(), df["graph_ok"].sum()],
    "Accuracy (tất cả)": [f"{flat_acc:.0%}", f"{graph_acc:.0%}"],
    "Accuracy (đa bước)": [f"{multi['flat_ok'].mean():.0%}", f"{multi['graph_ok'].mean():.0%}"],
})
summary''')

md(r"""### Các trường hợp Flat RAG **bị ảo giác** nhưng GraphRAG **trả lời đúng** (Deliverable yêu cầu)""")
code(r'''halluc = df[(~df["flat_ok"]) & (df["graph_ok"])]
print(f"Có {len(halluc)} trường hợp Flat RAG sai/ảo giác mà GraphRAG đúng:\n")
for _, r in halluc.iterrows():
    print("Q :", r["question"])
    print("  Flat RAG  (sai) :", r["flat_answer"])
    print("  GraphRAG  (đúng):", r["graph_answer"])
    print("  Đáp án          :", r["reference"], "\n")''')

# ---------------------------------------------------------------- Cost analysis
md(r"""## Phân tích chi phí (Deliverable #4) — Token usage & Time

So sánh chi phí giai đoạn **Indexing** (dựng đồ thị, chỉ làm 1 lần) và **Evaluation** (truy vấn).
Vì dùng Ollama cục bộ nên **chi phí tiền tệ = 0$**; ta đo token và thời gian như đại lượng thay thế.""")
code(r'''cost = pd.DataFrame([
    {"Giai đoạn": "Indexing (42 câu → triples)",
     "LLM calls": INDEX_STATS["calls"],
     "Prompt tokens": INDEX_STATS["prompt_tokens"],
     "Completion tokens": INDEX_STATS["completion_tokens"],
     "Tổng tokens": INDEX_STATS["prompt_tokens"] + INDEX_STATS["completion_tokens"],
     "Thời gian (s)": round(INDEX_SECONDS, 1)},
    {"Giai đoạn": "Evaluation (20 Q × 2 hệ thống)",
     "LLM calls": EVAL_STATS["calls"],
     "Prompt tokens": EVAL_STATS["prompt_tokens"],
     "Completion tokens": EVAL_STATS["completion_tokens"],
     "Tổng tokens": EVAL_STATS["prompt_tokens"] + EVAL_STATS["completion_tokens"],
     "Thời gian (s)": round(EVAL_STATS["seconds"], 1)},
])
print("Mô hình:", LLM_MODEL, "| Chi phí tiền tệ: $0.00 (Ollama cục bộ)")
cost''')

# ---------------------------------------------------------------- Persist results
md(r"""## Lưu kết quả ra file (để output không bị mất)

Mọi bảng kết quả được ghi xuống đĩa (`results_20q_comparison.csv`, `cost_analysis.csv`,
`RESULTS.md`). Nhờ vậy dù notebook bị **Clear Outputs** hay chạy lại, dữ liệu vẫn còn nguyên
trong các file này.""")
code(r'''# 1) Bảng so sánh 20 câu hỏi -> CSV
df.to_csv("results_20q_comparison.csv", index=False, encoding="utf-8-sig")
# 2) Bảng chi phí -> CSV
cost.to_csv("cost_analysis.csv", index=False, encoding="utf-8-sig")

# 3) Báo cáo tổng hợp dạng Markdown (Deliverable #3 + #4) -> RESULTS.md
lines = []
lines.append("# KẾT QUẢ LAB DAY 19 — GraphRAG vs Flat RAG\n")
lines.append(f"- Mô hình LLM: `{LLM_MODEL}` (Ollama, local) · Chi phí tiền tệ: **$0.00**")
lines.append(f"- Embeddings: all-MiniLM-L6-v2 · Vector DB: FAISS · Graph: NetworkX "
             f"({G.number_of_nodes()} node, {G.number_of_edges()} cạnh)\n")

lines.append("## Tổng hợp độ chính xác")
lines.append(summary.to_markdown(index=False))

lines.append("\n## Deliverable #4 — Phân tích chi phí (Token usage & Time)")
lines.append(cost.to_markdown(index=False))

lines.append("\n## Deliverable #3 — Bảng so sánh 20 câu hỏi")
tbl = df[["hop", "question", "flat_ok", "graph_ok", "flat_answer", "graph_answer"]]
lines.append(tbl.to_markdown(index=False))

lines.append("\n## Các trường hợp Flat RAG ảo giác nhưng GraphRAG đúng")
for _, r in halluc.iterrows():
    lines.append(f"\n**Q:** {r['question']}")
    lines.append(f"- Flat RAG (sai): {r['flat_answer']}")
    lines.append(f"- GraphRAG (đúng): {r['graph_answer']}")
    lines.append(f"- Đáp án: {r['reference']}")

with open("RESULTS.md", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))

print("Đã lưu: results_20q_comparison.csv, cost_analysis.csv, RESULTS.md")
print("\n----- RESULTS.md (preview) -----\n")
print("\n".join(lines[:14]))''')

# ---------------------------------------------------------------- Conclusion
md(r"""## Kết luận

- **GraphRAG vượt trội ở câu hỏi đa bước:** nhờ duyệt cạnh quan hệ tường minh (BFS 2-hop), nó ghép được
  các sự kiện rời rạc mà Flat RAG — vốn chỉ dựa trên độ tương tự vector — bỏ sót và hay **bịa**.
- **Khử trùng lặp** là điều kiện cần để các chuỗi suy luận không bị đứt gãy.
- **Chi phí:** Indexing tốn nhiều LLM call hơn (mỗi câu một lần trích xuất) nhưng chỉ làm **một lần**;
  truy vấn sau đó rẻ. Flat RAG rẻ khi index (chỉ embed) nhưng **trả giá bằng độ chính xác** ở câu đa bước.
- Với corpus lớn hơn, nên chuyển NetworkX → **Neo4j** (Cypher + Bloom) để truy vấn và trực quan hóa tốt hơn.

**Deliverables:** ✅ mã nguồn (notebook) · ✅ ảnh đồ thị `knowledge_graph.png` · ✅ bảng so sánh 20 câu hỏi ·
✅ phân tích token/time.
""")

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python", "version": "3.12"}
with open("Day19_GraphRAG.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print(f"Wrote Day19_GraphRAG.ipynb with {len(cells)} cells")
