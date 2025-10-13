import os
import sys
import pytest
import pandas as pd
import csv
from sentence_transformers import SentenceTransformer, util

# Ensure project root is on sys.path so 'RAG' package is importable when running the test directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from RAG.utils import pipeline_question
EVAL_FILE = os.path.join(os.path.dirname(__file__), "test_rag.csv")
import datetime

# Create a fresh timestamped results file for each run
TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_FILE = os.path.join(os.path.dirname(__file__), f"test_rag_results_{TS}.csv")



# Charger le modèle pour embeddings (CPU)
model = SentenceTransformer("embaas/sentence-transformers-multilingual-e5-base", device="cpu")

def text_similarity_embeddings(a: str, b: str) -> float:
    """
    Calcule la similarité cosine entre deux textes en utilisant des embeddings.
    Plus sémantique que TF-IDF, adapté aux textes français.

    a, b : str - textes à comparer
    return : float - similarité cosine entre 0 et 1
    """
    if not a or not b:
        return 0.0
    try:
        # Encoder les deux textes
        emb_a = model.encode(a, convert_to_tensor=True)
        emb_b = model.encode(b, convert_to_tensor=True)
        
        # Calculer la similarité cosine
        score = util.cos_sim(emb_a, emb_b)
        return float(score.item())
    except Exception as e:
        print("Erreur dans text_similarity_embeddings:", e)
        return 0.0


def _append_result(row: dict, generated: str, similarity: float):
    write_header = not os.path.exists(RESULT_FILE)
    keys = list(row.keys()) + ["generated_answer", "similarity"]
    with open(RESULT_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        if write_header:
            writer.writeheader()
        out = dict(row)
        out["generated_answer"] = generated
        out["similarity"] = f"{similarity:.4f}"
        writer.writerow(out)


@pytest.mark.parametrize("row", pd.read_csv(EVAL_FILE, sep=';', encoding='latin-1').to_dict(orient="records"))
def test_rag_response_minimal(row):
    """Minimal test: ensure pipeline_question returns a non-empty string for each question."""
    question = row.get("question")
    response = pipeline_question(question)

    print(f"\n---\nQuestion: {question}\nGot: {response}\n")

    assert isinstance(response, str)
    assert response.strip() != "", "pipeline_question must return non-empty text"

    # Compute similarity with expected answer if present and append result
    expected = row.get("expected_answer", "")
    sim = text_similarity_embeddings(response, expected)
    try:
        _append_result(row, response, sim)
    except Exception as e:
        # Do not fail the test because of IO; just warn
        print(f"Warning: failed to write result CSV: {e}")