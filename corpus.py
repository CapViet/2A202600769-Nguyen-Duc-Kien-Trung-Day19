"""
Tech Company Corpus + benchmark question set for the Day-19 GraphRAG lab.

The corpus is deliberately written so that many facts are *chained* across
sentences (e.g. DeepMind -> founded by Demis Hassabis; DeepMind -> acquired by
Google). Those chains are what GraphRAG can traverse but Flat RAG, which only
sees the top-k most similar passages, tends to miss -> hallucination.
"""

CORPUS = [
    # OpenAI cluster
    "OpenAI was founded by Sam Altman and Elon Musk in 2015.",
    "OpenAI is headquartered in San Francisco.",
    "OpenAI created the ChatGPT product.",
    "OpenAI developed the GPT-4 model.",
    "Microsoft invested in OpenAI in 2019.",
    "Sam Altman is the CEO of OpenAI.",

    # Google / Alphabet / DeepMind cluster
    "Google was founded by Larry Page and Sergey Brin in 1998.",
    "Google is headquartered in Mountain View.",
    "Alphabet is the parent company of Google.",
    "Sundar Pichai is the CEO of Google.",
    "Google acquired DeepMind in 2014.",
    "DeepMind was founded by Demis Hassabis in 2010.",
    "DeepMind is headquartered in London.",
    "DeepMind developed the AlphaGo program.",
    "Google acquired YouTube in 2006.",
    "YouTube was founded by Steve Chen and Chad Hurley in 2005.",
    "Google developed the TensorFlow framework.",
    "Google developed the Gemini model.",

    # Microsoft cluster
    "Microsoft was founded by Bill Gates and Paul Allen in 1975.",
    "Microsoft is headquartered in Redmond.",
    "Satya Nadella is the CEO of Microsoft.",
    "Microsoft acquired GitHub in 2018.",
    "Microsoft acquired LinkedIn in 2016.",
    "GitHub was founded by Tom Preston-Werner in 2008.",
    "Microsoft developed the Windows operating system.",

    # Meta cluster
    "Meta was founded by Mark Zuckerberg in 2004.",
    "Meta is headquartered in Menlo Park.",
    "Meta acquired Instagram in 2012.",
    "Meta acquired WhatsApp in 2014.",
    "Instagram was founded by Kevin Systrom in 2010.",
    "WhatsApp was founded by Jan Koum in 2009.",
    "Meta developed the LLaMA model.",

    # Nvidia / Tesla / SpaceX / Anthropic / Apple cluster
    "Nvidia was founded by Jensen Huang in 1993.",
    "Nvidia is headquartered in Santa Clara.",
    "Nvidia developed the CUDA platform.",
    "Tesla was founded by Elon Musk in 2003.",
    "Tesla is headquartered in Austin.",
    "SpaceX was founded by Elon Musk in 2002.",
    "Anthropic was founded by Dario Amodei in 2021.",
    "Anthropic developed the Claude model.",
    "Amazon invested in Anthropic in 2023.",
    "Apple was founded by Steve Jobs and Steve Wozniak in 1976.",
    "Tim Cook is the CEO of Apple.",
    "Apple is headquartered in Cupertino.",
]

# Benchmark questions. `hop` flags whether answering needs to combine facts
# from MORE THAN ONE sentence (the multi-hop / GraphRAG-favouring cases).
BENCHMARK = [
    # ---- single-hop (both systems should get these) ----
    {"q": "Who founded OpenAI?", "ref": "Sam Altman and Elon Musk", "hop": 1},
    {"q": "Where is DeepMind headquartered?", "ref": "London", "hop": 1},
    {"q": "Who is the CEO of Microsoft?", "ref": "Satya Nadella", "hop": 1},
    {"q": "In what year was Instagram founded?", "ref": "2010", "hop": 1},
    {"q": "Which company did Google acquire in 2006?", "ref": "YouTube", "hop": 1},
    {"q": "Who founded Nvidia?", "ref": "Jensen Huang", "hop": 1},
    {"q": "What model did Anthropic develop?", "ref": "Claude", "hop": 1},

    # ---- multi-hop (combine 2+ facts; Flat RAG tends to miss / hallucinate) ----
    {"q": "Who founded the company that Google acquired in 2014?",
     "ref": "Demis Hassabis (founder of DeepMind)", "hop": 2},
    {"q": "Which company acquired the company that developed AlphaGo?",
     "ref": "Google (acquired DeepMind)", "hop": 2},
    {"q": "Who founded the company that created ChatGPT?",
     "ref": "Sam Altman and Elon Musk", "hop": 2},
    {"q": "Which big company invested in the maker of GPT-4?",
     "ref": "Microsoft", "hop": 2},
    {"q": "Who is the CEO of the parent company's subsidiary that acquired YouTube?",
     "ref": "Sundar Pichai (CEO of Google)", "hop": 3},
    {"q": "Which company founded by Mark Zuckerberg acquired WhatsApp?",
     "ref": "Meta", "hop": 2},
    {"q": "Who founded the photo-sharing app that Meta acquired in 2012?",
     "ref": "Kevin Systrom (founder of Instagram)", "hop": 2},
    {"q": "Which code-hosting company that Microsoft acquired was founded by Tom Preston-Werner?",
     "ref": "GitHub", "hop": 2},
    {"q": "Elon Musk co-founded OpenAI; which car company did he also found?",
     "ref": "Tesla", "hop": 2},
    {"q": "Which company invested in the maker of the Claude model?",
     "ref": "Amazon (invested in Anthropic)", "hop": 2},
    {"q": "Who founded the company headquartered in London that Google owns?",
     "ref": "Demis Hassabis (DeepMind)", "hop": 2},
    {"q": "Name a founder shared by OpenAI, Tesla, and SpaceX.",
     "ref": "Elon Musk", "hop": 3},
    {"q": "Which company developed the model named Gemini and who is its CEO?",
     "ref": "Google, CEO Sundar Pichai", "hop": 2},
]
