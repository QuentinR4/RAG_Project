from utils import (
    pipeline_add_new_document,
    pipeline_question,
)
import argparse
import logging
import os
import time

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(docs=None, questions=None, force_reindex=False, k: int = 20):
    """Exécute les actions demandées.

    - docs: liste de chemins PDF à indexer (ou None)
    - questions: liste de questions à poser (ou None)
    - force_reindex: recrée l'index FAISS au lieu de l'étendre
    - k: nombre de documents renvoyés par le retriever
    """
    any_action = False

    # Indexation de documents
    if docs:
        for doc_path in docs:
            any_action = True
            logging.info(f"Ajout d'un document : {doc_path}")
            pipeline_add_new_document(doc_path, force_reindex)
            logging.info("Index mis à jour")

    # Questions
    if questions:
        for q in questions:
            any_action = True
            logging.info(f"Question : {q}")
            start_time = time.time()
            try:
                answer = pipeline_question(q, k=k)
                logging.info(f"Réponse : {answer}")
            finally:
                end_time = time.time()
                logging.info(f"Temps de réponse : {end_time - start_time:.2f} s")

    if not any_action:
        logging.warning("Aucune action demandée. Utilisez --doc et/ou --question. Voir --help.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CLI — indexer des PDF et poser des questions")
    parser.add_argument(
        "-d", "--doc", action="append", default=None,
        help="Chemin vers un PDF à indexer (répéter l'option pour plusieurs documents)",
    )
    parser.add_argument(
        "-q", "--question", action="append", default=None,
        help="Question à poser (répéter l'option pour plusieurs questions)",
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Forcer la recréation de l'index FAISS (écrase l'existant)",
    )
    parser.add_argument(
        "-k", type=int, default=20,
        help="Nombre de documents retournés par le retriever (k)",
    )

    args = parser.parse_args()

    # Petit rappel sur GEMINI_API_KEY si la personne va poser des questions
    if args.question and not os.getenv("GEMINI_API_KEY"):
        logging.warning("GEMINI_API_KEY n'est pas défini (les questions risquent d'échouer)")

    main(docs=args.doc, questions=args.question, force_reindex=args.force_reindex, k=args.k)

