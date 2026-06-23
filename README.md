# Lab Day 19 — GraphRAG trên *US Electric Vehicle Corpus*

**Sinh viên:** Nguyễn Đức Kiên Trung · **MSSV:** 2A202600769 · **Ngày:** 2026-06-23

Pipeline GraphRAG hoàn chỉnh (Indexing → Construction → Multi-hop Query → Evaluation)
trên **bộ dữ liệu thật do giảng viên cung cấp** (`dataset/` — 70 tài liệu web về ngành xe điện Mỹ),
so sánh với Flat RAG. Chạy **cục bộ, miễn phí** bằng Ollama + NetworkX (LLM provider có thể đổi).

## Stack
| Thành phần | Công cụ (mặc định) |
|---|---|
| LLM | Ollama · `qwen2.5:3b` (đổi được — xem dưới) |
| Embeddings | `sentence-transformers · all-MiniLM-L6-v2` |
| Vector index (Flat RAG) | FAISS (cosine) |
| Knowledge Graph (GraphRAG) | NetworkX MultiDiGraph |
| Visualization | Matplotlib |

## Đổi LLM provider (1 biến môi trường, không sửa code)
```bash
# Mặc định: cục bộ, miễn phí
GRAPHRAG_PROVIDER=ollama     GRAPHRAG_MODEL=qwen2.5:3b
# OpenAI
GRAPHRAG_PROVIDER=openai     GRAPHRAG_MODEL=gpt-4o-mini       OPENAI_API_KEY=...
# Anthropic Claude
GRAPHRAG_PROVIDER=anthropic  GRAPHRAG_MODEL=claude-opus-4-8   ANTHROPIC_API_KEY=...
# Google Gemini
GRAPHRAG_PROVIDER=google     GRAPHRAG_MODEL=gemini-1.5-flash  GOOGLE_API_KEY=...
```
Token usage & latency được đo cho mọi provider (phục vụ phân tích chi phí).

## Files
| File | Mô tả |
|---|---|
| `Day19_GraphRAG.ipynb` | **Notebook deliverable chính** (đã chạy sẵn, có output + bảng so sánh) |
| `corpus_loader.py` | Nạp / làm sạch / chunk 70 tài liệu EV (lọc boilerplate + nhiễu PDF) |
| `graphrag_core.py` | Trích xuất triple, dựng graph, query 2-hop, Flat RAG, **provider switch** |
| `benchmark_ev.py` | 20 câu hỏi benchmark (soạn theo facts có thật trong graph) |
| `run_indexing.py` | Bước indexing (chậm, **resumable**) → ghi `artifacts/` |
| `build_notebook.py` | Sinh notebook từ code đã kiểm thử |
| `artifacts/` | Kết quả indexing đã cache (triples, chunks, graph.graphml, stats) |
| `knowledge_graph.png` | Ảnh đồ thị tri thức (Deliverable #2) |
| `RESULTS.md`, `*.csv` | Bảng so sánh 20 câu + chi phí (bản lưu cố định) |

## Cách chạy
```bash
ollama pull qwen2.5:3b
pip install networkx matplotlib pandas faiss-cpu sentence-transformers nbformat jupyter tabulate

# 1) Indexing (1 lần, ~30 phút trên qwen2.5:3b; có checkpoint, chạy lại sẽ resume)
USE_TF=0 python -u run_indexing.py

# 2) Sinh + chạy notebook (nạp artifacts đã cache nên nhanh)
python build_notebook.py
USE_TF=0 jupyter nbconvert --to notebook --execute --inplace Day19_GraphRAG.ipynb
```
> `USE_TF=0` để tránh xung đột TensorFlow/Keras 3 khi import `sentence-transformers`.

## Ghi chú thiết kế
- **Lấy mẫu chunk (stride 5/doc):** indexing toàn bộ ~2000 chunk bằng mô hình cục bộ tốn hàng giờ;
  ta lấy mẫu trải đều mỗi tài liệu (341 chunk) — đánh đổi cost/coverage điển hình, được nêu trong phân tích chi phí.
- **So sánh công bằng:** Flat RAG và GraphRAG dùng **cùng tập chunk**; biến số duy nhất là cách truy hồi.
