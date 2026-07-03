#defino los modelos de datos a usar

from pydantic import BaseModel, Field

class QueryResponse(BaseModel):
    """Esquema para la respuesta final del pipeline RAG."""
    user_question: str = Field(
        ..., 
        description="La pregunta exacta que ingresó el usuario."
    )
    system_answer: str = Field(
        ..., 
        description="La respuesta generada por el LLM basándose únicamente en el contexto recuperado."
    )
    chunks_related: list[str] = Field(
        ..., 
        description="Lista de los fragmentos de texto (chunks) que se utilizaron como contexto."
    )

    
class EvaluationResult(BaseModel):
    """Esquema para la evaluación realizada por el Agente Evaluador."""
    score: int = Field(
        ..., 
        description="Puntuación de 0 a 10 sobre la calidad, relevancia y completitud de la respuesta.",
        ge=0,
        le=10
    )
    reason: str = Field(
        ..., 
        description="Justificación detallada del puntaje (mínimo 50 caracteres), explicando qué se cubrió y qué faltó."
    )

class IndexEntry(BaseModel):
    """Esquema para un elemento guardado en la base de datos vectorial (JSON)."""
    id: int = Field(..., description="Identificador único del chunk.")
    text: str = Field(..., description="El contenido del texto (fragmento).")
    embedding: list[float] = Field(..., description="El vector de embeddings generado para este texto.")    