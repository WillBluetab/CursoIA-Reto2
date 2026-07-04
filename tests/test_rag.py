import os
import sys
import unittest

# Añadir la carpeta 'src' al path de búsqueda de Python para permitir importaciones directas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from query import calculate_cosine_similarity
from schemas import IndexEntry, QueryResponse, EvaluationResult
from build_index import validate_chunks_tokens, split_into_chunks


class TestRAGPipeline(unittest.TestCase):
    """Pruebas unitarias para validar los componentes core del sistema RAG."""

    # -------------------------------------------------------------
    # 1. Pruebas para la Similitud Coseno
    # -------------------------------------------------------------
    def test_cosine_similarity_identical_vectors(self):
        """Dos vectores idénticos deben tener similitud coseno cercana a 1.0."""
        v1 = [1.0, 2.0, 3.0]
        v2 = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(calculate_cosine_similarity(v1, v2), 1.0, places=5)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Vectores perpendiculares u ortogonales deben tener similitud coseno de 0.0."""
        v1 = [1.0, 0.0]
        v2 = [0.0, 1.0]
        self.assertAlmostEqual(calculate_cosine_similarity(v1, v2), 0.0, places=5)

    def test_cosine_similarity_opposite_vectors(self):
        """Vectores opuestos deben tener similitud coseno cercana a -1.0."""
        v1 = [1.0, 1.0]
        v2 = [-1.0, -1.0]
        self.assertAlmostEqual(calculate_cosine_similarity(v1, v2), -1.0, places=5)

    def test_cosine_similarity_zero_vectors_graceful_handling(self):
        """Un vector de ceros no debe causar ZeroDivisionError, debe retornar 0.0."""
        v1 = [0.0, 0.0]
        v2 = [1.0, 2.0]
        try:
            similarity = calculate_cosine_similarity(v1, v2)
            self.assertEqual(similarity, 0.0)
        except ZeroDivisionError:
            self.fail("calculate_cosine_similarity lanzó ZeroDivisionError con un vector de ceros.")


    # -------------------------------------------------------------
    # 2. Pruebas para las Validaciones de Pydantic
    # -------------------------------------------------------------
    def test_index_entry_validation_success(self):
        """IndexEntry debe instanciarse con éxito con datos válidos."""
        data = {
            "id": 1,
            "text": "Texto de prueba para el chunk.",
            "embedding": [0.1, -0.2, 0.5]
        }
        entry = IndexEntry(**data)
        self.assertEqual(entry.id, 1)
        self.assertEqual(entry.text, "Texto de prueba para el chunk.")
        self.assertEqual(entry.embedding, [0.1, -0.2, 0.5])

    def test_index_entry_validation_failure_missing_field(self):
        """IndexEntry debe fallar si falta algún campo obligatorio (ej. embedding)."""
        data = {
            "id": 1,
            "text": "Texto de prueba sin embedding."
        }
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            IndexEntry(**data)

    def test_evaluation_result_validation_success(self):
        """EvaluationResult debe ser válido si la nota está entre 0 y 10."""
        data = {
            "score": 8,
            "reason": "La respuesta cubre de manera excelente la pregunta basándose en el contexto."
        }
        eval_res = EvaluationResult(**data)
        self.assertEqual(eval_res.score, 8)

    def test_evaluation_result_validation_failure_score_out_of_bounds(self):
        """EvaluationResult debe fallar si la nota es menor que 0 o mayor que 10."""
        from pydantic import ValidationError
        
        # Caso nota negativa
        with self.assertRaises(ValidationError):
            EvaluationResult(score=-1, reason="Justificación de longitud adecuada para cumplir los requisitos...")

        # Caso nota superior al máximo
        with self.assertRaises(ValidationError):
            EvaluationResult(score=11, reason="Justificación de longitud adecuada para cumplir los requisitos...")


    # -------------------------------------------------------------
    # 3. Pruebas para el Pipeline de Datos (Chunking y Fusión)
    # -------------------------------------------------------------
    def test_split_into_chunks(self):
        """split_into_chunks debe dividir un texto largo en fragmentos."""
        texto_largo = "Palabra " * 200  # Genera un texto largo
        chunks = split_into_chunks(texto_largo, chunk_size=100, chunk_overlap=10)
        self.assertTrue(len(chunks) > 1)

    def test_validate_chunks_tokens_consolidation(self):
        """validate_chunks_tokens debe consolidar (fusionar) chunks muy cortos con sus vecinos."""
        # Un chunk largo (tendrá unos 60-70 tokens) y un chunk extremadamente corto (4 tokens)
        chunk_largo = "Este es un fragmento de texto sustancialmente largo diseñado para superar el umbral mínimo de tokens configurado por el sistema de políticas internas."
        chunk_corto = "Muy corto."
        
        chunks_entrada = [chunk_largo, chunk_corto]
        
        # Mandamos min_tokens = 50. El primer chunk tiene > 50 tokens, el segundo tiene ~2 tokens.
        # El segundo debe unirse al anterior.
        chunks_consolidados = validate_chunks_tokens(chunks_entrada, min_tokens=30, max_tokens=100)
        
        # Deben haberse consolidado de 2 chunks iniciales a 1 solo chunk unificado
        self.assertEqual(len(chunks_consolidados), 1)
        self.assertIn(chunk_corto, chunks_consolidados[0])
        self.assertIn(chunk_largo, chunks_consolidados[0])


if __name__ == '__main__':
    unittest.main()
