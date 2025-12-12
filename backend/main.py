from fastapi import FastAPI
from ingest import ingest_all, retrieve
from ai_core_client import call_ai_core_llm

app = FastAPI()

# Auto-ingest on startup (optional)
@app.on_event("startup")
def startup_event():
    ingest_all()

@app.post("/query")
async def query(q: dict):
    question = q["question"]

    chunks = retrieve(question, k=5)

    context = "\n\n---\n\n".join(
        f"Source: {c['file']}\n{c['chunk']}" for c in chunks
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
