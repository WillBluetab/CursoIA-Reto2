import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Verificar la API Key
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Define OPENAI_API_KEY en el archivo .env")

# 1. Importar herramientas de LangChain
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

def load_documents_from_path(path: Path) -> list:
    """Routing por extensión: .pdf → PyPDFLoader, .txt/.md → TextLoader."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {path.resolve()}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(
            f"Extensión no soportada: {suffix!r}. Usa .pdf, .txt o .md."
        )

    return loader.load()

def load_documents_from_paths(paths: list[Path]) -> list:
    documents = []
    for path in paths:
        loaded = load_documents_from_path(path)
        documents.extend(loaded)
        print(f"  {path.name} ({path.suffix.lower()}) -> {len(loaded)} documento(s) LangChain")
    return documents

def main():
    print("--- 1. Cargando documentos ---")
    # Nota: El archivo sample_document.txt está en el directorio ../data/ (relativo a 'ejercicio')
    document_path = Path("../data/sample_document.txt")
    
    if not document_path.exists():
        # Fallback por si acaso
        document_path = Path("data/sample_document.txt")
        
    print(f"Buscando documento en: {document_path.resolve()}")
    documents = load_documents_from_paths([document_path])
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    print(f"Total objetos Document: {len(documents)}")
    print(f"Caracteres totales (aprox.): {total_chars}")
    
    print("\n--- 2. Dividiendo en chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Número de chunks creados: {len(chunks)}")
    if chunks:
        print(f"Vista previa del primer chunk:\n{chunks[0].page_content[:200]}...\n")

    print("--- 3. Inicializando embeddings y base vectorial (Chroma) ---")
    embeddings = OpenAIEmbeddings()
    persist_dir = "../chroma_db"
    
    # Crear e indexar
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    print(f"Base vectorial Chroma creada y guardada en: {persist_dir}")

    print("\n--- 4. Configurando el Retriever ---")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )

    print("\n--- 5. Configurando el modelo de lenguaje (LLM) ---")
    # Cambiamos "gpt-5-nano" (que no existe) por "gpt-4o-mini"
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    print(f"LLM configurado correctamente con el modelo: {llm.model_name}")

    print("\n--- 6. Creando la cadena RetrievalQA (RAG) ---")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    print("Cadena RAG lista para responder preguntas.")

    print("\n--- 7. Ejecutando consulta ---")
    query = "¿Qué es RAG y cómo funciona?"
    print(f"Pregunta: {query}")
    
    result = qa_chain.invoke({"query": query})
    
    print(f"\nRespuesta del modelo:\n{result['result']}")
    print(f"\nDocumentos de origen usados: {len(result['source_documents'])}")
    for i, doc in enumerate(result['source_documents'], start=1):
         print(f"  [{i}] (Líneas 1-150 de preview): {doc.page_content[:150].replace(chr(10), ' ')}...")

if __name__ == "__main__":
    main()
