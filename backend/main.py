from fastapi import FastAPI
from ingest import ingest_all, retrieve
from ai_core_client import call_ai_core_llm

app = FastAPI()


def is_comparison_question(q: str) -> bool:
    keywords = ["compare", "difference", "vs", "versus"]
    return any(k in q.lower() for k in keywords)


def extract_entities_for_comparison(q: str):
    entities = []
    if "ecobridge" in q:
        entities.append("EcoBridge")
    if "ohzone" in q:
        entities.append("OhZone")
    return entities


@app.on_event("startup")
def startup_event():
    ingest_all()


@app.get("/getData")
async def dataCAll():
    return "heloo data"


@app.post("/query")
async def query(q: dict):
    question = q["question"]

    # 🔹 Entity detection
    entities = extract_entities_for_comparison(question.lower())

    # 🔹 Primary semantic retrieval (RAW question – no spell correction)
    chunks = retrieve(question, k=10)

    # 🔹 Section / heading boosting
    heading_keywords = [
        "delivery methodology",
        "implementation approach",
        "deployment",
        "governance",
        "security"
    ]

    for kw in heading_keywords:
        if kw in question.lower():
            chunks.extend(retrieve(kw, k=3))

    # 🔹 Entity boosting (semantic)
    for ent in entities:
        chunks.extend(retrieve(ent, k=3))

    # 🔹 Deduplicate chunks
    seen = set()
    final_chunks = []
    for c in chunks:
        key = (c["file"], c["chunk"])
        if key not in seen:
            seen.add(key)
            final_chunks.append(c)

    # 🔹 Build context
    context = "\n\n---\n\n".join(
        f"Source: {c['file']}\n{c['chunk']}" for c in final_chunks
    )

    rag_prompt = f"""
You are an enterprise chatbot.
You MUST answer only using the context below.
Do not use outside knowledge.
If the answer is not in the context, say:
"I don't know based on the available local documents."

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and cite source file names if possible.
"""

    answer = call_ai_core_llm(rag_prompt)
    return {"answer": answer}

