import os
import json
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from schemas import IndexEntry


# Cargar las variables de entorno (.env)
load_dotenv()

# Configuración del modelo y rutas
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").lower()  # "openai" o "local"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DOCUMENT_PATH = "data/faq_document.txt"
INDEX_PATH = "data/faq_index.json"


# Configuración del modelo y rutas
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DOCUMENT_PATH = "data/faq_document.txt"
INDEX_PATH = "data/faq_index.json"

def load_document(file_path: str) -> str:
    """Carga el documento de texto plano utilizando codificación UTF-8."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo en la ruta: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    """Divide el texto en fragmentos (chunks) usando RecursiveCharacterTextSplitter.
    
    Esta estrategia es mejor que una división por caracteres fijos porque intenta
    mantener juntos párrafos y oraciones completas antes de cortar.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    # split_text devuelve una lista de strings
    chunks = splitter.split_text(text)
    return chunks

def validate_chunks_tokens(chunks: list[str], min_tokens: int = 50, max_tokens: int = 500) -> list[str]:
    """Valida que todos los chunks cumplan el rango de tokens.
    
    Si un chunk es inferior al mínimo (min_tokens), lo fusiona con el chunk anterior
    para no perder información ni contexto semántico.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    valid_chunks = []
    
    for i, chunk in enumerate(chunks):
        num_tokens = len(encoding.encode(chunk))
        
        if num_tokens < min_tokens:
            if len(valid_chunks) > 0:
                # Se fusiona con el último chunk válido
                valid_chunks[-1] = valid_chunks[-1] + "\n\n" + chunk
                print(f"Fusión: Chunk {i} ({num_tokens} tokens) unido al anterior por ser muy corto.")
            else:
                # Si es el primer chunk del documento, se agrega de todos modos para no perderlo
                valid_chunks.append(chunk)
        elif num_tokens > max_tokens:
            # Advertencia por si el splitter generó algo demasiado grande
            print(f"Advertencia: Chunk {i} supera el límite máximo con {num_tokens} tokens.")
            valid_chunks.append(chunk)
        else:
            valid_chunks.append(chunk)
            
    print(f"Total chunks iniciales: {len(chunks)} -> Chunks consolidados: {len(valid_chunks)}")
    return valid_chunks

#Quiero validar usando tanto los modelos de OpenAI, como un ejercicio en Local, por lo que
#defino 2 funciones para crear los embeddings, ya sea para llamar una o la otra.

def generate_embeddings_openai(chunks: list[str]) -> list[list[float]]:
    """Genera embeddings utilizando la API de OpenAI."""
    embeddings_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return embeddings_model.embed_documents(chunks)


def generate_embeddings_local(chunks: list[str]) -> list[list[float]]:
    """Genera embeddings de forma local utilizando SentenceTransformers.
    
    Usa importación perezosa (lazy import) para no exigir las librerías locales
    si se utiliza el proveedor cloud de OpenAI.
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError(
            "Error: Para utilizar embeddings locales debes instalar las dependencias requeridas.\n"
            "Ejecuta en tu terminal: pip install sentence-transformers torch"
        )
        
    print(f"Cargando modelo local '{LOCAL_EMBEDDING_MODEL}' en memoria (esto puede tardar unos segundos)...")
    embeddings_model = HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    return embeddings_model.embed_documents(chunks)


def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    """Función de enrutamiento que decide qué proveedor de embeddings utilizar."""
    if EMBEDDING_PROVIDER == "openai":
        print("Proveedor de Embeddings: OpenAI (Cloud)")
        return generate_embeddings_openai(chunks)
    elif EMBEDDING_PROVIDER == "local":
        print("Proveedor de Embeddings: Sentence-Transformers (Local)")
        return generate_embeddings_local(chunks)
    else:
        raise ValueError(
            f"Proveedor de embeddings '{EMBEDDING_PROVIDER}' no soportado. "
            "Usa 'openai' o 'local'."
        )

def save_index(chunks: list[str], embeddings: list[list[float]], output_path: str):
    """Guarda los fragmentos de texto y sus respectivos vectores en un archivo JSON estructurado."""
    output_data = []
    for i, (chunk, vector) in enumerate(zip(chunks, embeddings)):
        # Pydantic valida automáticamente los tipos al instanciar el objeto
        entry = IndexEntry(id=i, text=chunk, embedding=vector)
        # model_dump() convierte el objeto Pydantic en un diccionario estándar de Python
        output_data.append(entry.model_dump())
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"Índice vectorial guardado exitosamente en: {output_path} ({len(output_data)} registros)")


def main():
    print("--- Iniciando Data Pipeline de Indexación ---")
    
    # 1. Cargar el documento
    text = load_document(DOCUMENT_PATH)
    
    # 2. Dividir en chunks (ajustamos chunk_size para obtener ~20-30 chunks de ~100-200 tokens)
    chunks = split_into_chunks(text, chunk_size=800, chunk_overlap=100)
    
    # 3. Validar los límites de tokens exigidos por la rúbrica (50 - 500)
    valid_chunks = validate_chunks_tokens(chunks)
    
    if len(valid_chunks) < 20:
        raise ValueError(f"Se obtuvieron solo {len(valid_chunks)} chunks válidos. Ajusta chunk_size para obtener al menos 20.")
        
    # 4. Generar embeddings
    print("Generando embeddings en OpenAI...")
    embeddings = generate_embeddings(valid_chunks)
    
    # 5. Guardar base de conocimiento
    save_index(valid_chunks, embeddings, INDEX_PATH)
    print("--- Fin de la Indexación ---")

if __name__ == "__main__":
    main()
