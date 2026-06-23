# KẾT QUẢ LAB DAY 19 — GraphRAG vs Flat RAG (US EV Corpus)

- LLM: `qwen2.5:3b` (Ollama, local) · Chi phí tiền tệ: **$0.00**
- Corpus: 70 docs → 341 sampled chunks · Graph: NetworkX (579 node, 455 cạnh)

## Tổng hợp độ chính xác
| Hệ thống   |   Đúng / 20 | Accuracy (tất cả)   | Accuracy (đa bước)   |
|:-----------|------------:|:--------------------|:---------------------|
| Flat RAG   |          12 | 57%                 | 56%                  |
| GraphRAG   |          19 | 90%                 | 78%                  |

## Deliverable #4 — Chi phí (Token usage & Time)
| Giai đoạn                       |   LLM calls |   Prompt tokens |   Completion tokens |   Tổng tokens |   Thời gian (s) |
|:--------------------------------|------------:|----------------:|--------------------:|--------------:|----------------:|
| Indexing (341 chunks → triples) |         341 |          180137 |               17727 |        197864 |          1782   |
| Evaluation (20 Q × 2 hệ thống)  |          42 |           24321 |                 447 |         24768 |           112.4 |

## Deliverable #3 — Bảng so sánh 20 câu hỏi
|   hop | question                                                                                | flat_ok   | graph_ok   | flat_answer                                                                                                                                              | graph_answer                                                                                                                       |
|------:|:----------------------------------------------------------------------------------------|:----------|:-----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|
|     1 | What stock ticker does Tesla trade under?                                               | True      | True       | TSLA                                                                                                                                                     | Tesla trades as TSLA.                                                                                                              |
|     1 | Which sedan model does NIO make?                                                        | False     | True       | I don't know.                                                                                                                                            | Model ET7                                                                                                                          |
|     1 | Who is the CEO of NIO?                                                                  | False     | True       | I don't know.                                                                                                                                            | William Li is the CEO of NIO.                                                                                                      |
|     1 | What stock ticker does Polestar trade under?                                            | True      | True       | Polestar trades under the stock ticker PSNY.                                                                                                             | Polestar trades under the stock ticker PSNY.                                                                                       |
|     1 | In which city is Polestar based?                                                        | False     | True       | I don't know.                                                                                                                                            | Polestar is based in GOTHENBURG.                                                                                                   |
|     1 | In what year was Nikola founded?                                                        | False     | True       | I don't know.                                                                                                                                            | Nikola was founded in 2015.                                                                                                        |
|     1 | In which U.S. city is Nikola headquartered?                                             | False     | True       | I don't know.                                                                                                                                            | Nikola is headquartered in PHOENIX.                                                                                                |
|     1 | What stock ticker does VinFast trade under?                                             | True      | True       | VinFast trades under the stock ticker VFS.                                                                                                               | VinFast trades as VFS.                                                                                                             |
|     1 | In which country is BYD based?                                                          | True      | True       | BYD is based in China.                                                                                                                                   | BYD is based in China.                                                                                                             |
|     1 | What charging network does Tesla operate?                                               | True      | True       | Tesla operates a national network of Superchargers and destination chargers that are predominantly located within Metropolitan Statistical Areas (MSAs). | Tesla operates a network of Superchargers and Destination Chargers.                                                                |
|     2 | What stock ticker does the EV maker based in Gothenburg trade under?                    | False     | True       | I don't know.                                                                                                                                            | PSNY                                                                                                                               |
|     2 | Which company, founded in 2015 and headquartered in Phoenix, makes electric trucks?     | True      | True       | Nikola Corporation                                                                                                                                       | Nikola                                                                                                                             |
|     2 | What kind of trucks does the company headquartered in Phoenix produce?                  | True      | True       | Nikola Corporation produces hydrogen electric trucks and battery powered trucks headquartered in Phoenix.                                                | The company headquartered in Phoenix, Nikola, produces Hydrogen Electric Trucks and Battery Powered Trucks.                        |
|     2 | Which China-based company competes with Tesla in the EV market?                         | True      | True       | BYD competes with Tesla in the EV market.                                                                                                                | BYD competes with Tesla in the EV market.                                                                                          |
|     2 | Who is the chief executive of the company that makes the ET7?                           | False     | True       | I don't know.                                                                                                                                            | William Li is the CEO of NIO, which makes the ET7. Therefore, the chief executive of the company that makes the ET7 is William Li. |
|     2 | Which Chinese EV company did Elon Musk comment on?                                      | True      | True       | BYD                                                                                                                                                      | BYD                                                                                                                                |
|     2 | What stock ticker does the maker of the Model Y trade under?                            | False     | True       | XPEV                                                                                                                                                     | Tesla trades as TSLA.                                                                                                              |
|     1 | What was Tesla's reported US EV market share in Q1 2024?                                | True      | True       | Tesla's reported US EV market share in Q1 2024 was 51.3%.                                                                                                | Tesla's market share in Q1 2024 was 51.3%.                                                                                         |
|     1 | What is VinFast's annual delivery target for vehicles?                                  | True      | True       | VinFast's annual delivery target for vehicles is 80,000.                                                                                                 | VinFast's annual delivery target is 80,000 vehicles.                                                                               |
|     2 | In which city is Tesla's most productive manufacturing hub located?                     | True      | False      | Tesla's most productive manufacturing hub is located in Shanghai.                                                                                        | I don't know.                                                                                                                      |
|     2 | Besides the Model Y, which more affordable Tesla sedan became popular after price cuts? | False     | False      | The Model 3 sedan became popular after Tesla's aggressive price cuts throughout the year.                                                                | I don't know.                                                                                                                      |

## Các trường hợp Flat RAG sai nhưng GraphRAG đúng

**Q:** Which sedan model does NIO make?
- Flat RAG (sai): I don't know.
- GraphRAG (đúng): Model ET7
- Đáp án: ET7

**Q:** Who is the CEO of NIO?
- Flat RAG (sai): I don't know.
- GraphRAG (đúng): William Li is the CEO of NIO.
- Đáp án: William Li

**Q:** In which city is Polestar based?
- Flat RAG (sai): I don't know.
- GraphRAG (đúng): Polestar is based in GOTHENBURG.
- Đáp án: Gothenburg

**Q:** In what year was Nikola founded?
- Flat RAG (sai): I don't know.
- GraphRAG (đúng): Nikola was founded in 2015.
- Đáp án: 2015

**Q:** In which U.S. city is Nikola headquartered?
- Flat RAG (sai): I don't know.
- GraphRAG (đúng): Nikola is headquartered in PHOENIX.
- Đáp án: Phoenix

**Q:** What stock ticker does the EV maker based in Gothenburg trade under?
- Flat RAG (sai): I don't know.
- GraphRAG (đúng): PSNY
- Đáp án: PSNY

**Q:** Who is the chief executive of the company that makes the ET7?
- Flat RAG (sai): I don't know.
- GraphRAG (đúng): William Li is the CEO of NIO, which makes the ET7. Therefore, the chief executive of the company that makes the ET7 is William Li.
- Đáp án: William Li

**Q:** What stock ticker does the maker of the Model Y trade under?
- Flat RAG (sai): XPEV
- GraphRAG (đúng): Tesla trades as TSLA.
- Đáp án: TSLA