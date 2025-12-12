import os
from dotenv import load_dotenv
from gen_ai_hub.proxy.langchain.openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()  # Load .env first

def get_llm():
    client_id = os.getenv("AICORE_CLIENT_ID")
    client_secret = os.getenv("AICORE_CLIENT_SECRET")
    auth_url = os.getenv("AICORE_AUTH_URL")
    base_url = os.getenv("AICORE_BASE_URL")
    deployment_id = os.getenv("AICORE_DEPLOYMENT_ID")

    # Validate environment
    if not all([client_id, client_secret, auth_url, base_url, deployment_id]):
        raise ValueError("❌ Missing AI Core credentials in .env")

    return ChatOpenAI(
        deployment_id=deployment_id,
        model_name="gpt-4",
        temperature=0.0,
        max_tokens=2000
    )

def call_ai_core_llm(prompt: str):
    llm = get_llm()  # Create LLM only when needed
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
