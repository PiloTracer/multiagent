# ./app/multi_agent_app.py

import os
import pathlib
import requests

from langchain.agents import initialize_agent, Tool
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS

def load_vector_store(directory: str, openai_api_key: str):
    idx_file = pathlib.Path(directory) / "index.faiss"
    if not idx_file.exists():
        raise ValueError(f"No index.faiss found in {directory}. Did ingestion run?")
    vs = FAISS.load_local(
        directory,
        OpenAIEmbeddings(openai_api_key=openai_api_key),
        allow_dangerous_deserialization=True
    )
    return vs

def doc_search_tool(query: str) -> str:
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])

def general_tool(query: str) -> str:
    return f"General tool response: {query}"

def external_report_tool(query: str) -> str:
    """
    Sample external API tool:
    Fetch some dummy JSON, parse it, and return a short 'report'.
    You can adapt this to your real API calls.
    """
    try:
        # Example external API - JSONPlaceholder
        # This fetches a dummy to-do item
        resp = requests.get("https://jsonplaceholder.typicode.com/todos/1")
        resp.raise_for_status()
        data = resp.json()
        # Build a short 'report'
        return (
            f"External Report:\n"
            f"Fetched ID: {data.get('id')}\n"
            f"Title: {data.get('title')}\n"
            f"Completed: {data.get('completed')}\n"
        )
    except Exception as e:
        return f"Error fetching external report: {str(e)}"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY in env")

# Load FAISS index
vector_store = load_vector_store("./faiss_store", OPENAI_API_KEY)

# Define the LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# Define Tools (including our new external report tool)
tools = [
    Tool(
        name="DocSearch",
        func=doc_search_tool,
        description="Search in custom documents"
    ),
    Tool(
        name="GeneralTool",
        func=general_tool,
        description="General queries unrelated to docs"
    ),
    Tool(
        name="ExternalReportTool",
        func=external_report_tool,
        description=(
            "Use this tool whenever the user requests an external or new report, "
            "especially if they mention 'fetch an external report' or 'API'. "
            "This tool returns JSON-based data from an external source."
        )
    ),
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

if __name__ == "__main__":
    # Test
    q = "Give me a new external report about something"
    ans = agent.run(q)
    print("Agent says:", ans)
