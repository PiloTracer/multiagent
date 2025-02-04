# ./app/multi_agent_app.py
import os
import pathlib
import re
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
        resp = requests.get("https://jsonplaceholder.typicode.com/todos/1")
        resp.raise_for_status()
        data = resp.json()
        return (
            f"External Report:\n"
            f"Fetched ID: {data.get('id')}\n"
            f"Title: {data.get('title')}\n"
            f"Completed: {data.get('completed')}\n"
        )
    except Exception as e:
        return f"Error fetching external report: {str(e)}"

def telegram_message_tool(query: str) -> str:
    """
    Sends a message to a Telegram chat using the Telegram Bot API.
    The query should contain the message text.
    """
    try:
        BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
        CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
        if not BOT_TOKEN or not CHAT_ID:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in environment variables.")
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": CHAT_ID,
            "text": query
        }
        response = requests.post(url, json=payload)
        response.raise_for_status()
        if response.status_code == 200:
            return "Message successfully sent to Telegram!"
        else:
            return f"Failed to send message. Response: {response.text}"
    except Exception as e:
        return f"Error sending Telegram message: {str(e)}"

def direct_telegram_message_tool(input: str) -> str:
    """
    Parses natural language commands in various languages such as:
    "tell CHAT_ID message text", "dile CHAT_ID message text", etc.
    Returns the result of sending a Telegram message to the specified CHAT_ID.
    """
    pattern = re.compile(r'^(tell|dile|envia a)\s+(\S+)\s+(.+)', re.IGNORECASE)
    match = pattern.match(input)
    if match:
        chat_id = match.group(2)
        message = match.group(3)
        return send_telegram_message(chat_id, message)
    return "Input must be in the format: 'tell CHAT_ID message text' (or similar in your language)."

def send_telegram_message(chat_id: str, message: str):
    BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return "Message sent successfully!"
    else:
        return f"Failed to send message. Response: {response.text}"

def find_chat_id_by_name(name: str) -> str:
    """
    Queries Telegram's getUpdates API to find the chat_id for a given name.
    """
    try:
        BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not BOT_TOKEN:
            raise ValueError("Missing TELEGRAM_BOT_TOKEN in environment variables.")
        
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        response = requests.get(url)
        response.raise_for_status()
        updates = response.json().get("result", [])
        
        for update in updates:
            message = update.get("message", {})
            chat = message.get("chat", {})
            chat_id = chat.get("id")
            first_name = chat.get("first_name", "").lower()
            
            if name.lower() in first_name:
                return f"I have found the chat ID for {chat.get('first_name')}: {chat_id}"
        
        return f"No chat ID found for the name: {name}"
    except Exception as e:
        return f"Error finding chat ID: {str(e)}"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY in env")
if not TELEGRAM_BOT_TOKEN:
    raise ValueError("No TELEGRAM_BOT_TOKEN in env")
if not TELEGRAM_CHAT_ID:
    raise ValueError("No TELEGRAM_CHAT_ID in env")

# Load FAISS index
vector_store = load_vector_store("./faiss_store", OPENAI_API_KEY)

# Define the LLM
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)

# Define Tools (including our new FindChatIdByNameTool)
tools = [
    Tool(
        name="DocSearch",
        func=doc_search_tool,
        description=(
            "Search within the integrated document repository for information relevant to your query. "
            "Use this tool to locate specific details, facts, and text passages from the provided documents. "
            "It helps you quickly find content from the document database."
        )
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
    Tool(
        name="TelegramMessageTool",
        func=telegram_message_tool,
        description=(
            "Use this tool to send a message to a Telegram account. "
            "The input should be the message text you want to send."
        )
    ),
    Tool(
        name="DirectTelegramMessageTool",
        func=direct_telegram_message_tool,
        description=(
            "Send a message to a specific Telegram user using natural language. For example: "
            "'tell CHAT_ID message text' or 'dile CHAT_ID message text'."
        )
    ),
    Tool(
        name="FindChatIdByNameTool",
        func=find_chat_id_by_name,
        description=(
            "Use this tool to find the chat ID of a Telegram user by their name. "
            "For example: 'Please give me the chat ID for John Doe'."
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
    q = "Please give me the chat ID for john doe"
    ans = agent.run(q)
    print("Agent says:", ans)