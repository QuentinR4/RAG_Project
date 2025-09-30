from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import FAISS
import os
import google.generativeai as genai



    
# Fonction pour charger un PDF
def load_pdf(path):
    loader = PyMuPDFLoader(path)
    return loader.load()

# Fonction pour splitter les documents
def split_docs(documents, chunk_size=1000, chunk_overlap=200):
    #splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=["\n\n", "\n", ".", " ", ""])
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separators=["\n\n", "\n", " ", ""] 
)
    return splitter.split_documents(documents)

# Fonction pour créer les embeddings
def create_embeddings(model_name="BAAI/bge-base-en-v1.5"):
    return HuggingFaceEmbeddings(model_name=model_name)

# Fonction pour créer le vector store
def create_vectorstore(docs, embeddings):
    db = FAISS.from_documents(docs, embeddings)
    return db.as_retriever()

# Fonction pour créer le LLM
def get_llm():
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    return genai.GenerativeModel("gemini-2.0-flash")

# Fonction pour construire la chaîne RAG
def build_rag_chain(llm, retriever):
    # Gemini API ne s'intègre pas directement à LangChain RetrievalQA
    # On retourne simplement le modèle Gemini et le retriever pour usage manuel
    return llm, retriever

# Fonction pour poser une question
def ask_question(chain, question):
    llm, retriever = chain
    # Recherche contextuelle avec le retriever
    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])
    prompt = f"Contexte:\n{context}\n\nQuestion: {question}\nRéponds en français en utilisant uniquement le contexte."
    response = llm.generate_content(prompt)
    return response.text
