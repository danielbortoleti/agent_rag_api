from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent.rag.query import query_rag
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

# Inicializa o app FastAPI
app = FastAPI()

# Configura o middleware de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Substitua por ["http://localhost:3000"] no ambiente real
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo da requisição
class Question(BaseModel):
    message: str

# Rota principal
@app.post("/ask")
def ask_question(q: Question):
    answer = query_rag(q.message)
    return {"response": answer}
