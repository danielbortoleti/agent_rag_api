import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def query_rag(question: str, index_path: str = "agent/rag_index", k: int = 4):
    # Embeddings e índice
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # LLM (GPT-4 Turbo)
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

    # Prompt com engenharia de linguagem empática e acessível
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Você é um agente de inteligência artificial criado para orientar pais e mães sobre dúvidas comuns de saúde infantil, especialmente sintomas de ouvido, nariz e garganta.

Use uma linguagem **simples, acolhedora e empática** — como se estivesse explicando para alguém com pouco tempo e que não tem conhecimento técnico.

Baseie sua resposta apenas nas informações abaixo:

{context}

Pergunta: {question}

Resposta clara, sem jargões, com acolhimento:
"""
    )

    # RAG com prompt customizado
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    result = qa_chain(question)
    answer = result["result"]
    source_docs = result["source_documents"]

    # Formatando fontes
    sources = "\n\n".join(
        f"Fonte {i+1}: {doc.metadata.get('source', 'desconhecida')}" for i, doc in enumerate(source_docs)
    )

    return f"{answer}\n\n### Fontes consultadas:\n{sources}"
