from urllib import response
from utils import (
    pipeline_add_new_document,
    pipeline_question
)
import logging
import time
from langchain_community.vectorstores import FAISS

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(question=None, doc_path=None, force_reindex=False):

    if doc_path is not None:
        logging.info(f"Ajout d'un nouveau document : {doc_path}")
        pipeline_add_new_document(doc_path,force_reindex)
        logging.info("Document ajouté et index mis à jour.")  
    elif question is not None:
        logging.info(f"Question reçue : {question}")
        start_time = time.time()
        pipeline_question(question)
        end_time = time.time()
        logging.info(f"Temps de réponse : {end_time - start_time:.2f} secondes")
    

if __name__ == "__main__":
    # Pour forcer la recréation du cache, passe force_reindex=True
    #main(doc_path="./RAG/Dataset/HCC_RA_2025-18.07_web.pdf", force_reindex=True)
    #question = "De combien doit être l'objectif de rénovations performante de LLS par an  ?"
    question = "Comment évolue la température quotidienne de la mer ?"

    main(question)

