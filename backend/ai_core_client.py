import os
from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()


def get_llm():
    required = [
        "AICORE_CLIENT_ID",
        "AICORE_CLIENT_SECRET",
        "AICORE_AUTH_URL",
        "AICORE_BASE_URL",
        "AICORE_DEPLOYMENT_ID"
    ]

    for r in required:
        if not os.getenv(r):
            raise ValueError(f"Missing env variable: {r}")

    return ChatOpenAI(
        deployment_id=os.getenv("AICORE_DEPLOYMENT_ID"),
        model_name="gpt-4",
        temperature=0.0,
        max_tokens=2000
    )


def call_ai_core_llm(prompt: str) -> str:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
