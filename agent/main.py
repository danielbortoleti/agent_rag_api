from fastapi import FastAPI
from pydantic import BaseModel
from agent.rag.query import query_rag
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

class Question(BaseModel):
    message: str

@app.post("/ask")
def ask_question(q: Question):
    answer = query_rag(q.message)
    return {"response": answer}
