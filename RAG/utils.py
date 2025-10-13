from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
import os
import google.generativeai as genai
from .abbreviation import pipeline_abreviations
from .figures import save_identified_pages, analyze_saved_pages_with_gemini, load_figure_analyses
from langchain.schema import Document
import csv

    
# Fonction pour charger un PDF
def load_pdf(path):
    loader = PyMuPDFLoader(path)
    return loader.load()

# Fonction pour splitter les documents
def split_docs(documents, chunk_size=450, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=[". ", "? ","\n\n", "\n", ] )
    return splitter.split_documents(documents)

# Fonction pour créer les embeddings
def create_embeddings(model_name="embaas/sentence-transformers-multilingual-e5-base"):
    return HuggingFaceEmbeddings(model_name=model_name)

# Fonction pour créer le vector store
def create_vectorstore(docs, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

# Fonction pour créer le LLM
def get_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-2.5-flash-lite")

# Fonction pour construire la chaîne RAG
def build_rag_chain(llm, retriever):
    # Gemini API ne s'intègre pas directement à LangChain RetrievalQA
    # On retourne simplement le modèle Gemini et le retriever pour usage manuel
    return llm, retriever

# Fonction pour enrichir les chunks avec les définitions des abréviations
def enrich_chunks_with_abbreviations(chunks, abbr_dict):
    """
    Enrichit chaque chunk en ajoutant les définitions des abréviations présentes.
    
    chunks : list[Document] - les chunks déjà découpés
    abbr_dict : dict - dictionnaire {abréviation: définition}
    
    Retourne une liste de chunks enrichis (Document)
    """
    enriched_chunks = []

    for chunk in chunks:
        text = chunk.page_content
        added_definitions = []

        for abbr, definition in abbr_dict.items():
            # Ignorer si la définition est None ou vide
            if not definition:
                continue
            # Si l'abréviation apparaît dans le chunk, ajouter la définition entre parenthèses
            if abbr in text:
                # Vérifier si la définition n'est pas déjà présente
                if definition not in text:
                    text = text.replace(abbr, f"{abbr} ({definition})")
                    added_definitions.append(f"{abbr}: {definition}")

        enriched_chunks.append(Document(
            page_content=text,
            metadata=chunk.metadata
        ))

    return enriched_chunks

# Fonction pour poser une question
def ask_question(chain, question):
    llm, retriever = chain
    # Recherche contextuelle avec le retriever
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])            
    prompt = f"Contexte:\n{context}\n\nQuestion: {question}\n Si le texte contient des abréviations, explique-les à partir du contexte ou de tes connaissances générales. Réponds en français et de manière claire."
    response = llm.generate_content(prompt)
    return response.text


def pipeline_add_new_document(doc_path,force_reindex=False):
    file_name = os.path.basename(doc_path)
    figures_path="./RAG/Dataset/rag_figures/"+file_name
    cache_path = "./RAG/cache/faiss_index"
    
    # Charger le document
    documents = load_pdf(doc_path)
    print(f"Document chargé avec {len(documents)} pages.")

    # Découper le document
    docs = split_docs(documents)
    print(f"Document découpé en {len(docs)} chunks.")

    # Gestion des abréviations
    doc_abreviations = pipeline_abreviations(doc_path)
    if isinstance(doc_abreviations, dict):
        # Enrichir les chunks avec les abréviations
        docs = enrich_chunks_with_abbreviations(docs, doc_abreviations)
        print("Chunks enrichis avec les abréviations.")

    # Gestion des figures si existantes
    save_identified_pages(doc_path, figures_path, 15)
    analyze_saved_pages_with_gemini(figures_path)
    doc_figures=load_figure_analyses("./RAG/Dataset/rag_figures/_summary.json")

        
    # Fusionner les documents texte - figures
    all_docs = docs + doc_figures 
    print(f"Total de {len(all_docs)} documents (texte + figures + abréviations) à indexer.")

    # Créer les embeddings
    embeddings = create_embeddings()
    print("Embeddings créés.")
    
    # Créer ou mettre à jour le vector store
    if os.path.exists(cache_path) and not force_reindex:
        db = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
        print("Index FAISS existant chargé.")
        db.add_documents(all_docs)
        print(f"{len(all_docs)} nouveaux chunks ajoutés à l'index FAISS.")
    else:
        print("Création d'un nouvel index FAISS...")
        db = FAISS.from_documents(all_docs, embeddings)
        print("Nouveau index FAISS créé.")
    db.save_local(cache_path)
    print("Index FAISS sauvegardé localement.")
    
def pipeline_question(question, k: int = 20):
    cache_path = "./RAG/cache/faiss_index"
    if not os.path.exists(cache_path):
        raise ValueError("Le cache FAISS n'existe pas. Veuillez d'abord ajouter un document.")
    
    embeddings = create_embeddings()
    db = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
    print("Index FAISS existant chargé.")
    # Configure how many documents the retriever should return (k)
    retriever = db.as_retriever(search_kwargs={"k": k})
    print(f"Retriever configured to return k={k} documents")
    
    print("Initialisation du LLM Gemini...")
    llm = get_llm()
    print("LLM prêt.")

    print("Construction du pipeline RAG...")
    chain = build_rag_chain(llm, retriever)
    print("Pipeline RAG prêt.")

    print(f"Question posée : {question}")
    response = ask_question(chain, question)
    print(f"Réponse générée :\n{response}")
    return response


