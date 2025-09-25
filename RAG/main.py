from utils import load_pdf, split_docs, create_embeddings, create_vectorstore, get_llm, build_rag_chain, ask_question

import time
import os

def main(force_reindex=False):
	pdf_path = "./RAG/Dataset/HCC_RA_2025-18.07_web.pdf"
	#question = "De combien sont les émissions du secteur de l'industrie en 2024 ?"
	question = "De combien sont les émissions du secteur du bâtiment en 2024 ?"
	print("Chargement du PDF...")
	t0 = time.time()
	documents = load_pdf(pdf_path)
	print(f"PDF chargé ({len(documents)} pages) en {time.time()-t0:.2f}s")

	print("Découpage du texte...")
	t0 = time.time()
	docs = split_docs(documents)
	print(f"Texte découpé ({len(docs)} chunks) en {time.time()-t0:.2f}s")

	print("Création des embeddings...")
	t0 = time.time()
	embeddings = create_embeddings()
	print(f"Embeddings créés en {time.time()-t0:.2f}s")

	print("Chargement ou création du cache vectoriel FAISS...")
	t0 = time.time()
	cache_path = "./RAG/cache/faiss_index"
	from langchain_community.vectorstores import FAISS
	if os.path.exists(cache_path) and not force_reindex:
		print("Cache FAISS trouvé, chargement...")
		db = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
	else:
		print("Création ou recréation de l'index FAISS...")
		db = FAISS.from_documents(docs, embeddings)
		db.save_local(cache_path)
		print("Index FAISS sauvegardé dans le cache.")
	retriever = db.as_retriever()
	print(f"Base vectorielle prête en {time.time()-t0:.2f}s")

	print("Initialisation du LLM Gemini...")
	t0 = time.time()
	llm = get_llm()
	print(f"LLM prêt en {time.time()-t0:.2f}s")

	print("Construction du pipeline RAG...")
	chain = build_rag_chain(llm, retriever)
	print("Pipeline RAG prêt.")

	print(f"Question posée : {question}")
	t0 = time.time()
	response = ask_question(chain, question)
	print(f"Réponse générée en {time.time()-t0:.2f}s :\n{response}")

if __name__ == "__main__":
	# Pour forcer la recréation du cache, passe force_reindex=True
	main(force_reindex=False)
