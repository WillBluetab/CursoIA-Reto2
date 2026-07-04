import os
import json
import argparse
import numpy as np
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from schemas import IndexEntry, QueryResponse, EvaluationResult

# Cargar variables de entorno
load_dotenv()

INDEX_PATH = "data/faq_index.json"
# Configuración del modelo
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()  # "openai" o "local"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "phi3")  # O "llama3", "gemma2", etc.


def load_index(index_path: str) -> list[IndexEntry]:
    """Carga y valida los chunks y embeddings del índice JSON."""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No existe el índice en: {index_path}. Ejecuta build_index.py primero.")
    with open(index_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Validamos cada elemento usando Pydantic
    return [IndexEntry(**item) for item in data]

def embed_query(query_text: str) -> list[float]:
    """Genera el embedding para la consulta utilizando el proveedor configurado."""
    if EMBEDDING_PROVIDER == "openai":
        embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        return embeddings_model.embed_query(query_text)
    elif EMBEDDING_PROVIDER == "local":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError("Para usar embeddings locales instala: pip install sentence-transformers torch")
        embeddings_model = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        return embeddings_model.embed_query(query_text)
    else:
        raise ValueError(f"Proveedor '{EMBEDDING_PROVIDER}' no soportado.")

def calculate_cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Calcula la similitud coseno de forma explícita usando NumPy."""
    a = np.array(v1)
    b = np.array(v2)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

def search_similar_chunks(query_vector: list[float], index_data: list[IndexEntry], top_k: int = 3) -> list[IndexEntry]:
    """Realiza la búsqueda k-NN calculando la similitud con cada elemento del índice."""
    scored_chunks = []
    for entry in index_data:
        similarity = calculate_cosine_similarity(query_vector, entry.embedding)
        scored_chunks.append((similarity, entry))
    
    # Ordenar de mayor a menor similitud
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Retornar los top_k
    top_entries = [entry for _, entry in scored_chunks[:top_k]]
    return top_entries

def generate_answer(query_text: str, retrieved_chunks: list[IndexEntry]) -> str:
    """Ensambla el contexto y la consulta, y llama al LLM para generar la respuesta."""
    # Unimos los textos de los chunks para el contexto
    context = "\n\n---\n\n".join([chunk.text for chunk in retrieved_chunks])
    
    system_prompt = (
        "Eres un chatbot de soporte de HR para FAQs en la empresa TalentoHub.\n"
        "Responde a la pregunta del usuario utilizando únicamente el contexto proporcionado abajo.\n"
        "Si el contexto no tiene la información para responder, di amablemente: "
        "'Lo siento, no tengo esa información en mis políticas documentadas.'\n\n"
        f"Contexto disponible:\n{context}"
    )
    
    llm = get_llm()
    
    # Invocamos al modelo
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_text}
    ]
    response = llm.invoke(messages)
    return response.content

def evaluate_rag_response(query_text: str, answer_text: str, retrieved_chunks: list[IndexEntry]) -> EvaluationResult:
    """Agente Evaluador: Puntúa la respuesta de 0 a 10 con justificación estructurada."""
    context = "\n\n---\n\n".join([chunk.text for chunk in retrieved_chunks])
    
    eval_prompt = (
        "Eres un auditor experto en calidad de respuestas de sistemas RAG.\n"
        "Evalúa la calidad de la respuesta generada considerando tres dimensiones:\n"
        "1. Relevancia de los chunks respecto a la pregunta.\n"
        "2. Precisión de la respuesta (que no invente datos fuera del contexto).\n"
        "3. Completitud (que responda a toda la duda del usuario).\n\n"
        "ENTRADAS:\n"
        f"Pregunta del Usuario: {query_text}\n"
        f"Contexto Recuperado: {context}\n"
        f"Respuesta del Sistema: {answer_text}\n\n"
        "Instrucciones: Evalúa de 0 a 10 y escribe una justificación detallada de al menos 50 caracteres."
    )
    
    llm = get_llm()
    # Forzamos la salida estructurada de Pydantic usando Structured Outputs de OpenAI
    structured_llm = llm.with_structured_output(EvaluationResult)
    return structured_llm.invoke(eval_prompt)

def get_llm():
    """Retorna el modelo de lenguaje (LLM) configurado (OpenAI o Local)."""
    if LLM_PROVIDER == "openai":
        return ChatOpenAI(model=LLM_MODEL, temperature=0.0)
    elif LLM_PROVIDER == "local":
        try:
            # Importación perezosa
            from langchain_community.chat_models import ChatOllama
        except ImportError:
            raise ImportError(
                "Para utilizar un LLM local debes instalar la dependencia:\n"
                "pip install langchain-community"
            )
        print(f"Usando LLM Local en Ollama: '{LOCAL_LLM_MODEL}'")
        return ChatOllama(model=LOCAL_LLM_MODEL, temperature=0.0)
    else:
        raise ValueError(f"Proveedor de LLM '{LLM_PROVIDER}' no soportado.")


def main():
    parser = argparse.ArgumentParser(description="Query Pipeline RAG para TalentoHub")
    parser.add_argument("--query", type=str, required=True, help="La pregunta para el chatbot.")
    parser.add_argument("--top_k", type=int, default=3, help="Número de chunks a recuperar.")
    args = parser.parse_args()
    
    # 1. Cargar el índice
    index_data = load_index(INDEX_PATH)
    
    # 2. Convertir pregunta a vector
    query_vector = embed_query(args.query)
    
    # 3. Búsqueda vectorial
    relevant_chunks = search_similar_chunks(query_vector, index_data, top_k=args.top_k)
    
    # 4. Generar respuesta
    answer = generate_answer(args.query, relevant_chunks)
    
    # 5. Estructurar la respuesta final de consulta (Exactamente las 3 claves de la rúbrica)
    response_obj = QueryResponse(
        user_question=args.query,
        system_answer=answer,
        chunks_related=[chunk.text for chunk in relevant_chunks]
    )
    
    # Imprimir salida RAG en JSON formateado en consola
    print("\n=== RESPUESTA DEL CHATBOT (JSON) ===")
    print(json.dumps(response_obj.model_dump(), ensure_ascii=False, indent=2))
    
    # 6. Agente Evaluador (Puntúa la respuesta)
    print("\n=== EVALUACIÓN DEL AGENTE EVALUADOR ===")
    try:
        evaluation = evaluate_rag_response(args.query, answer, relevant_chunks)
        print(json.dumps(evaluation.model_dump(), ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Error al evaluar: {e}")

if __name__ == "__main__":
    main()
