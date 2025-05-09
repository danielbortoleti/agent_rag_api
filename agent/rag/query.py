import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

def query_rag(question: str, index_path: str = "agent/rag_index", k: int = 4):
    # Carregando embeddings e índice vetorial
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    # LLM configurada
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)

    # Prompt com linguagem simplificada, empática e acessível
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Você é um agente de inteligência artificial criado para orientar pais e mães que têm dúvidas sobre a saúde de seus filhos pequenos — principalmente quando o assunto é nariz entupido, dor de ouvido, garganta inflamada e coisas parecidas.

Sua missão é ajudar essas pessoas com muito carinho e respeito, usando uma linguagem **bem simples e fácil de entender**. Pense que está explicando para alguém que tem pouca escolaridade, está cansado e quer uma resposta rápida, direta e que traga segurança.

Evite palavras difíceis ou técnicas. Prefira exemplos do dia a dia, frases curtas, e explique tudo com calma e acolhimento — como se estivesse conversando com uma mãe ou um pai preocupado em casa.

Use apenas as informações confiáveis que estão no material abaixo. Não invente nada novo.

📌 Pergunta da pessoa:
{question}

📚 Informações que você pode usar:
{context}

💬 Resposta com carinho e em linguagem simples:
"""
    )

    # Construindo cadeia de RAG
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template}
    )

    # Executando a busca
    result = qa_chain(question)
    answer = result["result"]
    source_docs = result["source_documents"]

    # Processando e deduplicando fontes
    seen_sources = set()
    formatted_sources = []
    for i, doc in enumerate(source_docs):
        raw_path = doc.metadata.get("source", "desconhecida")
        clean_name = raw_path.split("/")[-1]  # remove "knowledge_base/" ou outros prefixos
        if clean_name not in seen_sources:
            seen_sources.add(clean_name)
            formatted_sources.append({
                "label": f"Fonte {len(formatted_sources) + 1}",
                "source": clean_name
            })

    # Formatando resposta para o front-end
    response = {
        "answer": answer.strip(),
        "sources": formatted_sources,
        "meta": {
            "modelo": "gpt-4-turbo",
            "documentos_consultados": len(formatted_sources),
            "index_path": index_path,
        }
    }

    return response
