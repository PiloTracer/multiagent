# ./app/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from multi_agent_app import agent

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    try:
        answer = agent.run(req.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
