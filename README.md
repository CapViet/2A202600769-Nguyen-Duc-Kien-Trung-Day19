# Lab Day 19 — GraphRAG với Tech Company Corpus

**Sinh viên:** Nguyễn Đức Kiên Trung · **MSSV:** 2A202600769 · **Ngày:** 2026-06-23

Pipeline GraphRAG hoàn chỉnh (Indexing → Construction → Multi-hop Query → Evaluation)
và so sánh với Flat RAG. Chạy **hoàn toàn cục bộ, miễn phí** bằng Ollama + NetworkX.

## Stack
| Thành phần | Công cụ |
|---|---|
| LLM | Ollama · `qwen2.5:3b` |
| Embeddings | `sentence-transformers · all-MiniLM-L6-v2` |
| Vector DB (Flat RAG) | FAISS |
| Knowledge Graph (GraphRAG) | NetworkX MultiDiGraph |
| Visualization | Matplotlib |

## Files
| File | Mô tả |
|---|---|
| `Day19_GraphRAG.ipynb` | **Notebook deliverable chính** (đã chạy sẵn, có output + bảng so sánh) |
| `graphrag_core.py` | Thư viện lõi: gọi Ollama, trích xuất triple, dựng graph, query 2-hop, Flat RAG |
| `corpus.py` | Tech Company Corpus (42 câu) + 20 câu hỏi benchmark |
| `build_notebook.py` | Script tạo notebook từ code đã kiểm thử |
| `knowledge_graph.png` | Ảnh đồ thị tri thức (Deliverable #2) |
| `RESULTS.md` | Báo cáo kết quả (bảng 20 câu + chi phí + ca ảo giác) — bản lưu cố định |
| `results_20q_comparison.csv` | Bảng so sánh 20 câu hỏi (Deliverable #3) dạng CSV |
| `cost_analysis.csv` | Phân tích token/time (Deliverable #4) dạng CSV |

## Cách chạy
```bash
# 1. Cài Ollama (https://ollama.com) rồi tải model
ollama pull qwen2.5:3b

# 2. Cài thư viện Python
pip install networkx matplotlib pandas faiss-cpu sentence-transformers nbformat jupyter

# 3. Mở và chạy notebook
jupyter notebook Day19_GraphRAG.ipynb
#   hoặc chạy đầu-cuối:
USE_TF=0 jupyter nbconvert --to notebook --execute --inplace Day19_GraphRAG.ipynb
```
> Đặt `USE_TF=0` trước khi import `sentence-transformers` để tránh xung đột TensorFlow/Keras 3.

## Kết quả chính
- GraphRAG duyệt cạnh quan hệ (BFS 2-hop) nên giải đúng các câu hỏi **đa bước**
  mà Flat RAG (chỉ dựa độ tương tự vector) bỏ sót và hay **bịa (hallucinate)**.
- Bảng so sánh 20 câu hỏi, danh sách trường hợp ảo giác, và phân tích token/time
  nằm trong notebook.
