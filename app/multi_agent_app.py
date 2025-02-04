# ./app/multi_agent_app.py
import os
import pathlib

# Remove the old import from `langchain_community.llms import OpenAI`.
# Instead, import the updated `OpenAI` class from `langchain_openai`.
# from langchain_community.llms import OpenAI
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

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY in env")

vector_store = load_vector_store("./faiss_store", OPENAI_API_KEY)

# Now import `OpenAI` from `langchain_openai` to avoid the deprecation warning.
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

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
    )
]

# The second warning about migrating to LangGraph is just a recommendation.
# If you are okay with continuing to use agents, you can ignore that warning or
# consider exploring LangGraph in the future.
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

if __name__ == "__main__":
    # Test
    q = "What do the docs say about deployment?"
    ans = agent.run(q)
    print("Agent says:", ans)
