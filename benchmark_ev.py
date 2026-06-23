"""
benchmark_ev.py
===============
20 benchmark questions for the US-EV corpus, authored AGAINST the actual
knowledge graph produced by run_indexing.py so every answer is grounded in
facts that really exist in the graph / chunks.

`hop`  = 1  single fact (answer adjacent to a named entity)
       >=2  multi-hop (must chain >=2 facts, e.g. place -> company -> ticker)
`ref`  short reference answer; grading checks its distinctive tokens appear
       in the system's answer.
"""

BENCHMARK = [
    # ---------------- single-hop ----------------
    {"q": "What stock ticker does Tesla trade under?", "ref": "TSLA", "hop": 1},
    {"q": "Which sedan model does NIO make?", "ref": "ET7", "hop": 1},
    {"q": "Who is the CEO of NIO?", "ref": "William Li", "hop": 1},
    {"q": "What stock ticker does Polestar trade under?", "ref": "PSNY", "hop": 1},
    {"q": "In which city is Polestar based?", "ref": "Gothenburg", "hop": 1},
    {"q": "In what year was Nikola founded?", "ref": "2015", "hop": 1},
    {"q": "In which U.S. city is Nikola headquartered?", "ref": "Phoenix", "hop": 1},
    {"q": "What stock ticker does VinFast trade under?", "ref": "VFS", "hop": 1},
    {"q": "In which country is BYD based?", "ref": "China", "hop": 1},
    {"q": "What charging network does Tesla operate?", "ref": "Superchargers", "hop": 1},

    # ---------------- multi-hop ----------------
    {"q": "What stock ticker does the EV maker based in Gothenburg trade under?",
     "ref": "PSNY", "hop": 2},
    {"q": "Which company, founded in 2015 and headquartered in Phoenix, makes electric trucks?",
     "ref": "Nikola", "hop": 2},
    {"q": "What kind of trucks does the company headquartered in Phoenix produce?",
     "ref": "Hydrogen", "hop": 2},
    {"q": "Which China-based company competes with Tesla in the EV market?",
     "ref": "BYD", "hop": 2},
    {"q": "Who is the chief executive of the company that makes the ET7?",
     "ref": "William Li", "hop": 2},
    {"q": "Which Chinese EV company did Elon Musk comment on?",
     "ref": "BYD", "hop": 2},
    {"q": "What stock ticker does the maker of the Model Y trade under?",
     "ref": "TSLA", "hop": 2},
    {"q": "What was Tesla's reported US EV market share in Q1 2024?",
     "ref": "51.3", "hop": 1},
    {"q": "What is VinFast's annual delivery target for vehicles?",
     "ref": "80,000", "hop": 1},

    # ---- coverage / robustness: facts present in the CORPUS but NOT cleanly
    # captured by the (sampled) extraction -> tests honest GraphRAG limitations
    {"q": "In which city is Tesla's most productive manufacturing hub located?",
     "ref": "Shanghai", "hop": 2},
    {"q": "Besides the Model Y, which more affordable Tesla sedan became popular after price cuts?",
     "ref": "Model 3", "hop": 2},
]
